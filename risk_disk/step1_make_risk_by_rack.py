import os
import json
import numpy as np
import pandas as pd

IN_DIR = "out"
OUT_DIR = "out"

INPUT_SUMMARY = os.path.join(IN_DIR, "rack_risk_summary.csv")
OUTPUT_RISK_BY_RACK = os.path.join(OUT_DIR, "risk_by_rack.csv")
OUTPUT_PARAMS = os.path.join(OUT_DIR, "strategy_params.json")

# -------- risk_score 定义（推荐：用 high_ratio）--------
RISK_SCORE_COL = "high_ratio"   # 从 rack_risk_summary.csv 取这一列作为 risk_score
CLIP_TO_01 = True              # 风险得分裁剪到 [0,1]

# -------- hard 策略阈值 τ 定义方式（二选一或都输出）--------
HARD_TOP_PCTS = [0.05, 0.10]   # top 5% / 10% 作为禁区
HARD_ABS_TAUS = [0.2]          # 或者绝对阈值 risk_score >= 0.2

# -------- soft 策略强度 α 扫描 --------
SOFT_ALPHAS = [0, 1, 2, 5, 10]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_SUMMARY)
    if "rack_id" not in df.columns:
        raise ValueError("rack_risk_summary.csv 缺少 rack_id")
    if RISK_SCORE_COL not in df.columns:
        raise ValueError(f"rack_risk_summary.csv 缺少 {RISK_SCORE_COL}，可用列: {list(df.columns)}")

    risk_by_rack = df[["rack_id", RISK_SCORE_COL]].rename(columns={RISK_SCORE_COL: "risk_score"}).copy()

    # 可选：裁剪/保证数值类型
    risk_by_rack["risk_score"] = pd.to_numeric(risk_by_rack["risk_score"], errors="coerce")
    if risk_by_rack["risk_score"].isna().any():
        raise ValueError("risk_score 存在 NaN，请检查输入 summary 计算过程")

    if CLIP_TO_01:
        risk_by_rack["risk_score"] = risk_by_rack["risk_score"].clip(0.0, 1.0)

    # 写出 risk_by_rack.csv
    risk_by_rack.to_csv(OUTPUT_RISK_BY_RACK, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote {OUTPUT_RISK_BY_RACK} with columns {list(risk_by_rack.columns)}")
    print("[INFO] risk_score describe:\n", risk_by_rack["risk_score"].describe())

    # 计算 hard(top%) 对应的 τ（便于仿真直接用 risk_score >= τ）
    taus_from_top = {}
    scores = risk_by_rack["risk_score"].values
    for p in HARD_TOP_PCTS:
        # top p => 阈值取 (1-p) 分位数
        tau = float(np.quantile(scores, 1.0 - p))
        taus_from_top[str(p)] = tau

    params = {
        "risk_by_rack_file": os.path.basename(OUTPUT_RISK_BY_RACK),
        "risk_score_definition": {
            "source_column": RISK_SCORE_COL,
            "meaning": "per-rack high-risk disk ratio (0~1)" if RISK_SCORE_COL == "high_ratio" else "custom",
            "clip_to_01": CLIP_TO_01
        },
        "hard_mode": {
            "tau_from_top_percent": {
                "top_percents": HARD_TOP_PCTS,
                "taus": taus_from_top,
                "rule": "forbid racks with risk_score >= tau"
            },
            "tau_absolute": {
                "taus": HARD_ABS_TAUS,
                "rule": "forbid racks with risk_score >= tau"
            }
        },
        "soft_mode": {
            "alphas": SOFT_ALPHAS,
            "note": "alpha controls penalty/weight strength in placement decision (higher = more risk-averse)"
        }
    }

    with open(OUTPUT_PARAMS, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"[INFO] wrote {OUTPUT_PARAMS}")

    # 额外：打印一下 top% 禁区会禁掉多少 rack（sanity check）
    for p, tau in taus_from_top.items():
        forbidden = int((risk_by_rack["risk_score"] >= tau).sum())
        print(f"[CHECK] top {float(p)*100:.1f}% => tau={tau:.4f}, forbidden_racks={forbidden}/{len(risk_by_rack)}")

    for tau in HARD_ABS_TAUS:
        forbidden = int((risk_by_rack["risk_score"] >= tau).sum())
        print(f"[CHECK] abs tau={tau:.4f}, forbidden_racks={forbidden}/{len(risk_by_rack)}")


if __name__ == "__main__":
    main()