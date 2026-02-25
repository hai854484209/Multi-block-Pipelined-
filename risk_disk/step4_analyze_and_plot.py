# risk_disk/step4_analyze_and_plot.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None


# -----------------------------
# 默认输入/输出（可被命令行覆盖）
# -----------------------------
DEFAULT_IN_DIR = os.path.join("out", "step3")   # 里面应包含 step3_summary.csv 和 placements_*.csv
DEFAULT_OUT_DIR = os.path.join("out", "step4")

# 两个主要目标（越小越好）：
# 1) risky_disk_rate_among_observed 风险盘占比
# 2) rack_util_p99 负载集中度（热点）
OBJ_RISK = "risky_disk_rate_among_observed"
OBJ_HOT  = "rack_util_p99"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        type=str,
        default=DEFAULT_IN_DIR,
        help="Step3 output directory; must contain step3_summary.csv and placements_*.csv",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory for Step4 results",
    )
    return ap.parse_args()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def make_tag(top_pct: float, alpha: float) -> str:
    # 必须与 Step3 一致
    return f"top{int(top_pct * 100)}_alpha{str(alpha).replace('.', '_')}"


def pareto_front_minimize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    返回 Pareto 前沿（全是“越小越好”的目标）。
    一个点被支配：存在另一个点在所有目标 <= 且至少一个目标 <。
    """
    X = df[cols].to_numpy()
    n = X.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue

        dominates_i = np.all(X <= X[i], axis=1) & np.any(X < X[i], axis=1)
        if np.any(dominates_i):
            is_pareto[i] = False
            continue

        dominated_by_i = np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1)
        is_pareto[dominated_by_i] = False

    return df.loc[is_pareto].copy()


def normalize_0_1(series: pd.Series) -> pd.Series:
    mn, mx = float(series.min()), float(series.max())
    if mx - mn < 1e-12:
        return series * 0.0
    return (series - mn) / (mx - mn)


def plot_heatmap(pivot: pd.DataFrame, title: str, out_path: str, fmt: str = ".4f"):
    plt.figure(figsize=(8, 4.8), dpi=160)
    if sns is not None:
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap="viridis")
    else:
        ax = plt.gca()
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(x) for x in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(y) for y in pivot.index])

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i, j]:{fmt}}",
                        ha="center", va="center", color="white", fontsize=7)

    plt.title(title)
    plt.xlabel("alpha")
    plt.ylabel("top_pct")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_rack_risk_metrics_from_placements(placements_path: str) -> dict:
    """
    placements_{tag}.csv 每行一个成功副本，至少包含：
    obj_id, replica_id, rack_id, slot_id, rack_risk_score, forbidden_by_hard
    """
    pl = pd.read_csv(placements_path)

    need = {"obj_id", "rack_risk_score", "forbidden_by_hard"}
    missing = [c for c in need if c not in pl.columns]
    if missing:
        raise ValueError(f"{placements_path} 缺少列: {missing}")

    # 1) 副本平均 rack 风险
    replica_risk_mean = float(pl["rack_risk_score"].mean()) if len(pl) else np.nan

    # 2) 对象维度：每个 obj 的副本里取 max 风险，再对 obj 平均
    obj_max = pl.groupby("obj_id", as_index=True)["rack_risk_score"].max()
    object_max_risk_mean = float(obj_max.mean()) if len(obj_max) else np.nan

    # 3) 副本落在 hard forbidden 的比例（理论上 Step3 hard 已禁用，应接近 0）
    replica_in_forbidden_ratio = float(pl["forbidden_by_hard"].mean()) if len(pl) else np.nan

    return {
        "replica_risk_mean": replica_risk_mean,
        "object_max_risk_mean": object_max_risk_mean,
        "replica_in_forbidden_ratio": replica_in_forbidden_ratio,
        "placements_rows": int(len(pl)),
        "objects_observed": int(pl["obj_id"].nunique()) if len(pl) else 0
    }


def attach_rack_risk_metrics(ok_df: pd.DataFrame, step3_dir: str) -> pd.DataFrame:
    """
    对 ok_df 每个 (top_pct, alpha) 读取对应 placements_{tag}.csv，
    计算 rack-risk 三指标并合并回 ok_df。
    若 placements 文件不存在，则填 NaN 并给出 WARN。
    """
    rows = []
    for _, r in ok_df[["top_pct", "alpha"]].drop_duplicates().iterrows():
        top_pct = float(r["top_pct"])
        alpha = float(r["alpha"])
        tag = make_tag(top_pct, alpha)
        path = os.path.join(step3_dir, f"placements_{tag}.csv")

        if not os.path.exists(path):
            rows.append({
                "top_pct": top_pct,
                "alpha": alpha,
                "replica_risk_mean": np.nan,
                "object_max_risk_mean": np.nan,
                "replica_in_forbidden_ratio": np.nan,
                "placements_rows": 0,
                "objects_observed": 0,
                "_placements_missing": True
            })
            continue

        m = compute_rack_risk_metrics_from_placements(path)
        m.update({
            "top_pct": top_pct,
            "alpha": alpha,
            "_placements_missing": False
        })
        rows.append(m)

    mdf = pd.DataFrame(rows)
    out = ok_df.merge(mdf, on=["top_pct", "alpha"], how="left")
    return out


def main():
    args = parse_args()

    step3_dir = args.in_dir
    out_dir = args.out_dir

    ensure_dir(out_dir)

    in_summary = os.path.join(step3_dir, "step3_summary.csv")

    if not os.path.exists(in_summary):
        raise FileNotFoundError(
            f"找不到 {in_summary}。请确认 --in_dir 指向 Step3 输出目录（里面应包含 step3_summary.csv）"
        )

    df = pd.read_csv(in_summary)

    # 基本检查：确保 step3 跑完且数据完整
    required = ["top_pct", "alpha", "placed_replicas", "failed_replicas",
                "cluster_utilization", OBJ_RISK, OBJ_HOT]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"step3_summary 缺少列: {missing}")

    # 只保留成功放置的方案（若以后做高负载，可能会出现失败）
    ok = df[df["failed_replicas"] == 0].copy()

    # 新增：合并 rack-risk 三指标（来自 placements）
    ok = attach_rack_risk_metrics(ok, step3_dir)

    # -------- 1) 热力图：风险率 / p99 热点 --------
    pivot_risk = ok.pivot_table(index="top_pct", columns="alpha", values=OBJ_RISK, aggfunc="mean")
    pivot_hot  = ok.pivot_table(index="top_pct", columns="alpha", values=OBJ_HOT,  aggfunc="mean")

    plot_heatmap(
        pivot_risk,
        f"Heatmap: {OBJ_RISK} (lower is better)",
        os.path.join(out_dir, "heatmap_risky_rate.png"),
        fmt=".4f",
    )
    plot_heatmap(
        pivot_hot,
        f"Heatmap: {OBJ_HOT} (lower is better)",
        os.path.join(out_dir, "heatmap_rack_util_p99.png"),
        fmt=".4f",
    )

    # -------- 2) Pareto 前沿：在 (risk, hot) 两目标都最小 --------
    pareto = pareto_front_minimize(ok, [OBJ_RISK, OBJ_HOT])

    pareto_out = pareto.sort_values([OBJ_RISK, OBJ_HOT]).reset_index(drop=True)
    pareto_out.to_csv(os.path.join(out_dir, "pareto_front.csv"), index=False, encoding="utf-8-sig")

    # -------- 3) 散点图 + Pareto 曲线 --------
    plt.figure(figsize=(6.8, 5.2), dpi=160)
    plt.scatter(ok[OBJ_HOT], ok[OBJ_RISK], s=40, alpha=0.7, label="candidates")
    plt.scatter(pareto_out[OBJ_HOT], pareto_out[OBJ_RISK], s=80, label="Pareto", edgecolors="k")

    p_line = pareto_out.sort_values(OBJ_HOT)
    plt.plot(p_line[OBJ_HOT], p_line[OBJ_RISK], linewidth=1.5)

    for _, r in pareto_out.iterrows():
        plt.text(r[OBJ_HOT], r[OBJ_RISK], f"t={r['top_pct']},a={r['alpha']}",
                 fontsize=7, ha="left", va="bottom")

    plt.xlabel(OBJ_HOT + " (lower is better)")
    plt.ylabel(OBJ_RISK + " (lower is better)")
    plt.title("Trade-off: hotspot vs risk (Pareto front)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pareto_scatter.png"))
    plt.close()

    # -------- 4) 推荐排序（不改变你原来的两目标排序逻辑）--------
    # 仍然只用 (risk, hot) 计算 score，保证和旧版本可比
    w_risk = 0.7
    w_hot = 0.3
    ok["risk_n"] = normalize_0_1(ok[OBJ_RISK])
    ok["hot_n"] = normalize_0_1(ok[OBJ_HOT])
    ok["score"] = w_risk * ok["risk_n"] + w_hot * ok["hot_n"]  # 越小越好

    ranked = ok.sort_values("score").reset_index(drop=True)
    ranked.to_csv(os.path.join(out_dir, "step4_ranked.csv"), index=False, encoding="utf-8-sig")

    # 若 placements 缺失，给出提示（不阻断）
    if "_placements_missing" in ok.columns and ok["_placements_missing"].any():
        miss_n = int(ok["_placements_missing"].sum())
        print(f"[WARN] {miss_n} 个参数组合找不到 placements_{{tag}}.csv；rack-risk 三指标将为 NaN。")

    print("[INFO] wrote:", os.path.join(out_dir, "heatmap_risky_rate.png"))
    print("[INFO] wrote:", os.path.join(out_dir, "heatmap_rack_util_p99.png"))
    print("[INFO] wrote:", os.path.join(out_dir, "pareto_scatter.png"))
    print("[INFO] wrote:", os.path.join(out_dir, "pareto_front.csv"))
    print("[INFO] wrote:", os.path.join(out_dir, "step4_ranked.csv"))
    print("[DONE] step4")


if __name__ == "__main__":
    main()