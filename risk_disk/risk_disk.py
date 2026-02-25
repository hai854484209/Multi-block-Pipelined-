# risk_inject_rule01_empirical_out.py
import os
import numpy as np
import pandas as pd

# -----------------------------
# 配置
# -----------------------------
CSV_PATH = "disk_30d_features.csv"
OUT_DIR = "out"
ID_COL = "serial_number"

# 仿真规模
RACKS = 50
SLOTS_PER_RACK = 60
RANDOM_SEED = 42

# Step1：规则定义
RULE_MODE = "any_smart_flag"  # "any_smart_flag" 或 "thresholds"
THRESH = {
    "smart_187_ever": 1,
    "smart_188_ever": 1,
    "smart_197_ever": 1,
    "smart_198_ever": 1,
    "smart_5_ever": 1,
    "smart_199_ever": 1,
}

# Step2：注入方式（无聚集 empirical）
INJECT_MODE = "ratio"  # "ratio"（Bernoulli比例注入） 或 "bootstrap"
# -----------------------------


def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_risk_disk_rule01(df: pd.DataFrame) -> pd.Series:
    """
    输出 risk_disk ∈ {0,1}
    - any_smart_flag: 任意 *_ever > 0 即高风险
    - thresholds: 按 THRESH 指定列>=阈值即高风险（OR 逻辑）
    """
    if RULE_MODE == "any_smart_flag":
        cols = [c for c in df.columns if c.endswith("_ever")]
        if not cols:
            raise ValueError("找不到 *_ever 列，无法使用 any_smart_flag 规则。")
        risk = (df[cols].fillna(0).sum(axis=1) > 0).astype(int)

    elif RULE_MODE == "thresholds":
        cond = np.zeros(len(df), dtype=bool)
        for c, t in THRESH.items():
            if c not in df.columns:
                raise ValueError(f"阈值规则列不存在: {c}")
            cond |= (df[c].fillna(0) >= t)
        risk = cond.astype(int)

    else:
        raise ValueError("RULE_MODE must be 'any_smart_flag' or 'thresholds'")

    return risk


def simulate_racks(n_racks: int, slots_per_rack: int) -> pd.DataFrame:
    """构造 rack-slot 拓扑表"""
    return pd.DataFrame(
        [(r, s) for r in range(n_racks) for s in range(slots_per_rack)],
        columns=["rack_id", "slot_id"]
    )


def inject_empirical_by_ratio(rack_df: pd.DataFrame, p_high: float, rng: np.random.RandomState) -> pd.DataFrame:
    """方式A：只保持整体高风险比例 p_high（独立同分布）"""
    out = rack_df.copy()
    out["risk_disk"] = rng.binomial(n=1, p=p_high, size=len(out)).astype(int)
    return out


def inject_empirical_by_bootstrap(rack_df: pd.DataFrame, risk_disk: np.ndarray, rng: np.random.RandomState) -> pd.DataFrame:
    """方式B：从历史 risk_disk(0/1) 经验分布 bootstrap 抽样（同样是 i.i.d.）"""
    out = rack_df.copy()
    out["risk_disk"] = rng.choice(risk_disk.astype(int), size=len(out), replace=True).astype(int)
    return out


def summarize_rack(rack_assign: pd.DataFrame) -> pd.DataFrame:
    """每个 rack 的高风险数量与比例"""
    agg = rack_assign.groupby("rack_id").agg(
        high_cnt=("risk_disk", "sum"),
        high_ratio=("risk_disk", "mean"),
    ).reset_index()
    return agg


def main():
    ensure_out_dir(OUT_DIR)
    rng = np.random.RandomState(RANDOM_SEED)

    # -----------------------------
    # Load
    # -----------------------------
    df = pd.read_csv(CSV_PATH)
    if ID_COL not in df.columns:
        raise ValueError(f"missing column: {ID_COL}")

    # -----------------------------
    # Step 1: disk risk (0/1)
    # -----------------------------
    df = df.copy()
    df["risk_disk"] = compute_risk_disk_rule01(df)

    out_disk = os.path.join(OUT_DIR, "disk_30d_with_risk_rule01.csv")
    df.to_csv(out_disk, index=False, encoding="utf-8-sig")

    p_high = float(df["risk_disk"].mean())
    print(f"[INFO] disk risk_disk ratio = {p_high:.4f} ({int(df['risk_disk'].sum())}/{len(df)})")
    print(f"[INFO] wrote: {out_disk}")

    # -----------------------------
    # Step 2: inject to rack (empirical, no clustering)
    # -----------------------------
    rack_df = simulate_racks(RACKS, SLOTS_PER_RACK)

    if INJECT_MODE == "ratio":
        rack_assign = inject_empirical_by_ratio(rack_df, p_high, rng)
    elif INJECT_MODE == "bootstrap":
        rack_assign = inject_empirical_by_bootstrap(rack_df, df["risk_disk"].values, rng)
    else:
        raise ValueError("INJECT_MODE must be 'ratio' or 'bootstrap'")

    out_assign = os.path.join(OUT_DIR, "rack_risk_assignment.csv")
    rack_assign.to_csv(out_assign, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote: {out_assign}")

    # summary
    rack_summary = summarize_rack(rack_assign)
    out_summary = os.path.join(OUT_DIR, "rack_risk_summary.csv")
    rack_summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote: {out_summary}")

    # quick check
    print("[CHECK] rack overall high-risk ratio =", float(rack_assign["risk_disk"].mean()))
    print("[CHECK] rack summary head:\n", rack_summary.head(10))


if __name__ == "__main__":
    main()