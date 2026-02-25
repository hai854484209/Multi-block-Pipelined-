# risk_disk/step5_multi_seed_eval.py
import os
import math
import subprocess
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

# 保持与当前文件同目录运行（你原来就这么做）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join("out", "step5")

OBJ_RISK = "risky_disk_rate_among_observed"
OBJ_HOT = "rack_util_p99"

EXTRA_METRICS = [
    "replica_risk_mean",
    "object_max_risk_mean",
    "replica_in_forbidden_ratio",
]

W_RISK = 0.7
W_HOT = 0.3
Z_95 = 1.96


@dataclass
class Step3RunnerConfig:
    step3_script: Optional[str] = None
    python_exe: str = "python"


@dataclass
class Step4RunnerConfig:
    step4_script: Optional[str] = None
    python_exe: str = "python"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def normalize_0_1(series: pd.Series) -> pd.Series:
    mn, mx = float(series.min()), float(series.max())
    if mx - mn < 1e-12:
        return series * 0.0
    return (series - mn) / (mx - mn)


def pareto_front_minimize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
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


def run_step3_for_seed(cfg: Step3RunnerConfig, seed: int, out_dir: str):
    """
    需要 Step3 支持:
      --out_dir <dir>
      --placement_seed <seed>
    """
    if cfg.step3_script is None:
        raise ValueError("未配置 Step3RunnerConfig.step3_script，无法 auto_run_step3。")
    ensure_dir(out_dir)
    cmd = [cfg.python_exe, cfg.step3_script, "--out_dir", out_dir, "--placement_seed", str(seed)]
    print("[INFO] run:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_step4_for_seed(cfg: Step4RunnerConfig, in_dir: str, out_dir: str):
    """
    需要 Step4 支持:
      --in_dir <step3_seed_dir>
      --out_dir <step4_seed_dir>
    """
    if cfg.step4_script is None:
        raise ValueError("未配置 Step4RunnerConfig.step4_script，无法 auto_run_step4。")
    ensure_dir(out_dir)
    cmd = [cfg.python_exe, cfg.step4_script, "--in_dir", in_dir, "--out_dir", out_dir]
    print("[INFO] run:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def resolve_step3_summary_path(step3_root: str, seed: int, layout: str) -> str:
    if layout == "seed_dirs":
        return os.path.join(step3_root, f"seed_{seed}", "step3_summary.csv")
    elif layout == "flat":
        return os.path.join(step3_root, f"step3_summary_seed{seed}.csv")
    else:
        raise ValueError(f"unknown layout: {layout}")


def read_one_seed_metrics(step3_root: str, seed: int, layout: str) -> pd.DataFrame:
    path = resolve_step3_summary_path(step3_root, seed, layout)
    if not os.path.exists(path):
        hint = []
        if os.path.exists(step3_root):
            hint.append(f"[HINT] 当前 step3_root={step3_root} 内容示例：")
            try:
                items = sorted(os.listdir(step3_root))[:30]
                hint.append("  " + "\n  ".join(items))
            except Exception:
                pass
        hint.append("[HINT] 你需要为每个 seed 生成独立的 step3_summary：")
        hint.append("  layout=seed_dirs: out/step3/seed_0/step3_summary.csv ...")
        hint.append("  或 layout=flat:    out/step3/step3_summary_seed0.csv ...")
        raise FileNotFoundError(f"missing: {path}\n" + "\n".join(hint))

    df = pd.read_csv(path)
    df["seed"] = seed
    if "failed_replicas" in df.columns:
        df = df[df["failed_replicas"] == 0].copy()
    return df


def compute_score(df: pd.DataFrame) -> pd.Series:
    risk_n = normalize_0_1(df[OBJ_RISK])
    hot_n = normalize_0_1(df[OBJ_HOT])
    return W_RISK * risk_n + W_HOT * hot_n


def agg_mean_std_ci(x: pd.Series) -> pd.Series:
    x = x.dropna()
    n = int(x.shape[0])
    if n == 0:
        return pd.Series({"n": 0, "mean": np.nan, "std": np.nan, "ci95_low": np.nan, "ci95_high": np.nan})
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n >= 2 else 0.0
    se = std / math.sqrt(n) if n >= 2 else 0.0
    half = Z_95 * se
    return pd.Series({"n": n, "mean": mean, "std": std, "ci95_low": mean - half, "ci95_high": mean + half})


def load_step4_ranked(step4_seed_dir: str) -> pd.DataFrame:
    path = os.path.join(step4_seed_dir, "step4_ranked.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing: {path}；请确认 Step4 已按 seed 输出到该目录。")
    df = pd.read_csv(path)

    # Step4 ranked 必须包含这些 key 和三列指标
    need_cols = ["top_pct", "alpha", "tau"] + EXTRA_METRICS
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少列: {missing}")

    # 只保留 join 所需列，避免重复列污染
    return df[["top_pct", "alpha", "tau"] + EXTRA_METRICS].drop_duplicates(subset=["top_pct", "alpha", "tau"])


def main(
    seeds: List[int] = None,
    step3_root: str = os.path.join("out", "step3"),
    step4_root: str = os.path.join("out", "step4"),
    layout: str = "seed_dirs",
    auto_run_step3: bool = False,
    auto_run_step4: bool = True,
    runner_cfg: Step3RunnerConfig = Step3RunnerConfig(step3_script=None),
    runner4_cfg: Step4RunnerConfig = Step4RunnerConfig(step4_script=None),
):
    ensure_dir(OUT_DIR)

    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    per_seed_frames = []
    for s in seeds:
        seed_step3_dir = os.path.join(step3_root, f"seed_{s}")
        seed_step4_dir = os.path.join(step4_root, f"seed_{s}")

        if auto_run_step3:
            # auto-run 时强制使用 seed_dirs 输出，否则会覆盖
            run_step3_for_seed(runner_cfg, s, seed_step3_dir)

        if auto_run_step4:
            run_step4_for_seed(runner4_cfg, in_dir=seed_step3_dir, out_dir=seed_step4_dir)

        df_s = read_one_seed_metrics(step3_root, s, layout)
        df_s = df_s.copy()
        df_s["score"] = compute_score(df_s)

        # 合并 Step4 的三指标到每个 seed 的 df_s
        step4_pick = load_step4_ranked(seed_step4_dir)
        df_s = df_s.merge(step4_pick, on=["top_pct", "alpha", "tau"], how="left")

        per_seed_frames.append(df_s)

    long_df = pd.concat(per_seed_frames, ignore_index=True)

    # ----------------------------
    # 第二步关键：修正聚合键
    # 只按 (top_pct, alpha, tau) 聚合，保证每组输出 1 行
    # ----------------------------
    param_cols = [c for c in ["top_pct", "alpha", "tau"] if c in long_df.columns]
    if param_cols != ["top_pct", "alpha", "tau"]:
        raise ValueError(f"long_df 缺少聚合键列，当前 param_cols={param_cols}，需要 top_pct/alpha/tau 三列都存在。")

    # seed-level pareto + best
    long_df["_is_pareto_seed"] = False
    for s in seeds:
        sub = long_df[long_df["seed"] == s].copy()
        pf = pareto_front_minimize(sub, [OBJ_RISK, OBJ_HOT])
        long_df.loc[pf.index, "_is_pareto_seed"] = True

    long_df["_is_best_seed"] = False
    for s in seeds:
        sub = long_df[long_df["seed"] == s].copy()
        if not sub.empty:
            long_df.loc[sub["score"].idxmin(), "_is_best_seed"] = True

    long_out = os.path.join(OUT_DIR, "step5_seed_metrics_long.csv")
    long_df.to_csv(long_out, index=False, encoding="utf-8-sig")

    # 聚合指标：risk/hot + 三列（如果在 long_df 里）
    metric_cols = [OBJ_RISK, OBJ_HOT] + [m for m in EXTRA_METRICS if m in long_df.columns]

    g = long_df.groupby(["top_pct", "alpha", "tau"], dropna=False)

    agg_parts = []
    for m in metric_cols:
        stat = g[m].apply(agg_mean_std_ci).unstack()
        stat = stat.rename(columns={
            "n": f"{m}__n",
            "mean": f"{m}__mean",
            "std": f"{m}__std",
            "ci95_low": f"{m}__ci95_low",
            "ci95_high": f"{m}__ci95_high",
        })
        agg_parts.append(stat)

    pareto_freq = g["_is_pareto_seed"].mean().rename("pareto_freq")
    best_freq = g["_is_best_seed"].mean().rename("best_freq")

    agg_df = pd.concat(agg_parts + [pareto_freq, best_freq], axis=1).reset_index()

    agg_out = os.path.join(OUT_DIR, "step5_agg_metrics.csv")
    agg_df.to_csv(agg_out, index=False, encoding="utf-8-sig")

    print("[INFO] wrote:", long_out)
    print("[INFO] wrote:", agg_out)
    print("[DONE] step5")


if __name__ == "__main__":
    main(
        seeds=[0, 1, 2, 3, 4],
        step3_root=os.path.join("out", "step3"),
        step4_root=os.path.join("out", "step4"),
        layout="seed_dirs",
        auto_run_step3=True,
        auto_run_step4=True,
        runner_cfg=Step3RunnerConfig(
            step3_script=r"step3_simulate_placement_p2c.py",
            python_exe=r"D:\Pycharm\Pycharm date\pythonProject3\.venv\Scripts\python.exe",
        ),
        runner4_cfg=Step4RunnerConfig(
            step4_script=r"step4_analyze_and_plot.py",
            python_exe=r"D:\Pycharm\Pycharm date\pythonProject3\.venv\Scripts\python.exe",
        ),
    )