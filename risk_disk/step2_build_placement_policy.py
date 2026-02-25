import os
import json
import numpy as np
import pandas as pd

IN_DIR = "out"
OUT_DIR = os.path.join("out", "step2")

RISK_BY_RACK = os.path.join(IN_DIR, "risk_by_rack.csv")
PARAMS_JSON = os.path.join(IN_DIR, "strategy_params.json")


def load_params(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tau_from_top_percent(scores: np.ndarray, p: float) -> float:
    # top p% => threshold at (1-p) quantile
    return float(np.quantile(scores, 1.0 - p))


def build_hard_forbidden(df_risk: pd.DataFrame, p: float):
    tau = tau_from_top_percent(df_risk["risk_score"].values, p)
    forbidden = df_risk[df_risk["risk_score"] >= tau].copy()
    allowed = df_risk[df_risk["risk_score"] < tau].copy()
    return tau, forbidden, allowed


def build_soft_scores(df_allowed: pd.DataFrame, alpha: float):
    out = df_allowed.copy()
    out["alpha"] = float(alpha)
    out["cost"] = float(alpha) * out["risk_score"]

    # 若你后续想做“按权重随机选 rack”，用 weight 更方便
    # weight 越大越优先
    out["weight"] = np.exp(-out["cost"])

    # 排序：cost 小优先；若 cost 相同，再按 risk_score 小优先
    out = out.sort_values(["cost", "risk_score", "rack_id"], ascending=[True, True, True]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df_risk = pd.read_csv(RISK_BY_RACK)
    if not {"rack_id", "risk_score"}.issubset(df_risk.columns):
        raise ValueError(f"{RISK_BY_RACK} must contain rack_id, risk_score")

    df_risk = df_risk.copy()
    df_risk["risk_score"] = pd.to_numeric(df_risk["risk_score"], errors="coerce")
    if df_risk["risk_score"].isna().any():
        raise ValueError("risk_score has NaN")

    params = load_params(PARAMS_JSON)

    top_pcts = params["hard_mode"]["tau_from_top_percent"]["top_percents"]
    alphas = params["soft_mode"]["alphas"]

    # 对每个 top p% 生成：禁区 + (alpha 扫描的) 打分表
    for p in top_pcts:
        p = float(p)
        tau, forbidden, allowed = build_hard_forbidden(df_risk, p)

        # 1) hard forbidden racks
        forbid_path = os.path.join(OUT_DIR, f"hard_forbidden_racks_top{int(p*100)}.csv")
        forbidden_out = forbidden[["rack_id", "risk_score"]].copy()
        forbidden_out["tau"] = tau
        forbidden_out.to_csv(forbid_path, index=False, encoding="utf-8-sig")

        # 2) allowed list（有时也很有用）
        allow_path = os.path.join(OUT_DIR, f"hard_allowed_racks_top{int(p*100)}.csv")
        allowed_out = allowed[["rack_id", "risk_score"]].copy()
        allowed_out["tau"] = tau
        allowed_out.to_csv(allow_path, index=False, encoding="utf-8-sig")

        print(f"[INFO] top {p*100:.1f}% => tau={tau:.4f}, forbidden={len(forbidden)}/{len(df_risk)}, allowed={len(allowed)}")
        print(f"[INFO] wrote {forbid_path}")
        print(f"[INFO] wrote {allow_path}")

        # 3) soft scoring tables for each alpha
        for a in alphas:
            a = float(a)
            scored = build_soft_scores(allowed, a)

            score_path = os.path.join(
                OUT_DIR, f"rack_scores_top{int(p*100)}_alpha{str(a).replace('.', '_')}.csv"
            )
            scored.to_csv(score_path, index=False, encoding="utf-8-sig")
            print(f"[INFO] wrote {score_path}")

    # 写一份索引，方便后续 step3/4/5 自动发现文件
    index = {
        "risk_by_rack": os.path.relpath(RISK_BY_RACK, start="out"),
        "generated_in": OUT_DIR,
        "hard_top_pcts": top_pcts,
        "soft_alphas": alphas,
        "files": sorted(os.listdir(OUT_DIR))
    }
    index_path = os.path.join(OUT_DIR, "step2_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"[INFO] wrote {index_path}")


if __name__ == "__main__":
    main()