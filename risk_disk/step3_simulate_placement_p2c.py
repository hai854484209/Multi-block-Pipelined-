import os
import json
import argparse
import numpy as np
import pandas as pd

# -----------------------------
# 输入文件（来自 step1/step2）
# -----------------------------
RISK_BY_RACK = os.path.join("out", "risk_by_rack.csv")
STEP2_INDEX = os.path.join("out", "step2", "step2_index.json")

# 如果存在 rack-slot 风险（可选，用于统计落盘风险；不影响放置）
RACK_SLOT_RISK = os.path.join("out", "rack_risk_assignment.csv")  # columns: rack_id, slot_id, risk_disk(0/1)

# -----------------------------
# 默认输出目录（可被 --out_dir 覆盖）
# -----------------------------
DEFAULT_OUT_DIR = os.path.join("out", "step3")

# -----------------------------
# 仿真参数（可通过 --target_util 间接改变 N_OBJECTS）
# -----------------------------
N_OBJECTS = 20000
REPLICATION = 3
OBJ_SIZE_MB = 10
ENFORCE_DISTINCT_RACK = True

N_RACKS = 50
SLOTS_PER_RACK = 60
SLOT_CAPACITY_MB = 5000

MAX_TRIES_PER_REPLICA = 200
P2C_K = 2

# 是否输出 placements 明细（建议 True，供 Step4 计算 rack-risk 指标）
DUMP_PLACEMENTS = True


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def weighted_choice(rng: np.random.RandomState, items: np.ndarray, weights: np.ndarray):
    """按权重抽一个 item（weights 可不归一化）"""
    w = weights.astype(float)
    s = w.sum()
    if s <= 0:
        idx = rng.randint(0, len(items))
        return items[idx]
    p = w / s
    idx = rng.choice(len(items), p=p)
    return items[idx]


def make_cluster_topology(n_racks: int, slots_per_rack: int):
    rack_ids = np.arange(n_racks, dtype=int)
    slots = pd.DataFrame(
        [(r, s) for r in rack_ids for s in range(slots_per_rack)],
        columns=["rack_id", "slot_id"]
    )
    return rack_ids, slots


def init_capacity_state(slots_df: pd.DataFrame, slot_capacity_mb: int):
    st = slots_df.copy()
    st["cap_mb"] = float(slot_capacity_mb)
    st["used_mb"] = 0.0
    return st


def p2c_pick_slot(rng: np.random.RandomState, slot_rows: pd.DataFrame, obj_size_mb: float, k: int = 2):
    """
    P2C：在可容纳 obj 的 slots 中随机采样 k 个，选 used_mb 最小的那个。
    返回 slot_id 或 None
    """
    feasible = slot_rows[slot_rows["cap_mb"] - slot_rows["used_mb"] >= obj_size_mb]
    if len(feasible) == 0:
        return None

    if len(feasible) <= k:
        best = feasible.sort_values(["used_mb", "slot_id"], ascending=[True, True]).iloc[0]
        return int(best["slot_id"])

    sample_idx = rng.choice(feasible.index.values, size=k, replace=False)
    sample = feasible.loc[sample_idx]
    best = sample.sort_values(["used_mb", "slot_id"], ascending=[True, True]).iloc[0]
    return int(best["slot_id"])


def run_one_setting(
    rng: np.random.RandomState,
    top_pct: float,
    alpha: float,
    risk_by_rack: pd.DataFrame,
    rack_ids: np.ndarray,
    slots_state: pd.DataFrame,
    slot_risk_map: pd.DataFrame | None,
    dump_placements: bool = True,
    target_util: float | None = None,
):
    """
    单次实验（给定 top_pct, alpha）：
    - hard: forbid top_pct racks (allowed: score < tau)
    - soft: weight=exp(-alpha*risk_score) within allowed
    - place N_OBJECTS objects with REPLICATION replicas using P2C inside rack
    """
    # --- hard threshold from top pct ---
    risk_s = risk_by_rack.set_index("rack_id").loc[rack_ids, "risk_score"]
    scores = risk_s.values.astype(float)
    tau = float(np.quantile(scores, 1.0 - top_pct))

    allowed_mask = scores < tau
    allowed_racks = rack_ids[allowed_mask]
    allowed_scores = scores[allowed_mask]

    if len(allowed_racks) == 0:
        raise RuntimeError("No allowed racks under hard policy; check top_pct or risk_score distribution")

    weights = np.exp(-float(alpha) * allowed_scores)

    placed_replicas = 0
    failed_replicas = 0
    failed_objects = 0

    # disk-risk stats (optional)
    placed_on_risky_disk = 0
    total_placed_with_disk_risk_obs = 0

    placements = [] if dump_placements else None

    st = slots_state.copy()

    rack_to_idx = {int(r): st.index[st["rack_id"] == r].values for r in rack_ids}

    risk_lookup = None
    if slot_risk_map is not None:
        risk_lookup = slot_risk_map.set_index(["rack_id", "slot_id"])["risk_disk"].to_dict()

    for obj_id in range(N_OBJECTS):
        used_racks_for_obj = set()
        ok_replicas = 0

        for rep in range(REPLICATION):
            placed = False

            for _try in range(MAX_TRIES_PER_REPLICA):
                rack = int(weighted_choice(rng, allowed_racks, weights))

                if ENFORCE_DISTINCT_RACK and rack in used_racks_for_obj:
                    continue

                idxs = rack_to_idx[rack]
                slot_rows = st.loc[idxs, ["slot_id", "cap_mb", "used_mb"]]
                slot_id = p2c_pick_slot(rng, slot_rows, OBJ_SIZE_MB, k=P2C_K)
                if slot_id is None:
                    continue

                # commit
                row_idx = st.index[(st["rack_id"] == rack) & (st["slot_id"] == slot_id)][0]
                st.at[row_idx, "used_mb"] = float(st.at[row_idx, "used_mb"] + OBJ_SIZE_MB)

                used_racks_for_obj.add(rack)
                ok_replicas += 1
                placed_replicas += 1
                placed = True

                # disk risk stats
                if risk_lookup is not None:
                    rk = risk_lookup.get((rack, slot_id), None)
                    if rk is not None:
                        total_placed_with_disk_risk_obs += 1
                        placed_on_risky_disk += int(rk)

                if dump_placements:
                    rack_risk_score = float(risk_s.loc[rack])
                    forbidden_by_hard = int(rack_risk_score >= tau)
                    placements.append({
                        "obj_id": int(obj_id),
                        "replica_id": int(rep),
                        "rack_id": int(rack),
                        "slot_id": int(slot_id),
                        "rack_risk_score": rack_risk_score,
                        "tau": float(tau),
                        "forbidden_by_hard": forbidden_by_hard
                    })

                break

            if not placed:
                failed_replicas += 1

        if ok_replicas < REPLICATION:
            failed_objects += 1

    st["free_mb"] = st["cap_mb"] - st["used_mb"]
    util = float(st["used_mb"].sum() / st["cap_mb"].sum())

    rack_util = st.groupby("rack_id", as_index=False)[["used_mb", "cap_mb"]].sum()
    rack_util["util"] = rack_util["used_mb"] / rack_util["cap_mb"]
    rack_util = rack_util[["rack_id", "util"]]

    util_p50 = float(rack_util["util"].median())
    util_p90 = float(np.quantile(rack_util["util"].values, 0.90))
    util_p99 = float(np.quantile(rack_util["util"].values, 0.99))

    summary = {
        "top_pct": float(top_pct),
        "alpha": float(alpha),
        "tau": float(tau),

        # 关键：写入“实验设置”的 target_util（用于 FigA 横轴）
        "target_util": float(target_util) if target_util is not None else np.nan,

        "allowed_racks": int(len(allowed_racks)),
        "forbidden_racks": int(len(rack_ids) - len(allowed_racks)),
        "placed_replicas": int(placed_replicas),
        "failed_replicas": int(failed_replicas),
        "failed_objects": int(failed_objects),
        "cluster_utilization": float(util),
        "rack_util_p50": util_p50,
        "rack_util_p90": util_p90,
        "rack_util_p99": util_p99,
        "placed_on_risky_disk": int(placed_on_risky_disk),
        "placed_with_disk_risk_obs": int(total_placed_with_disk_risk_obs),
        "risky_disk_rate_among_observed": (
            float(placed_on_risky_disk) / total_placed_with_disk_risk_obs
            if total_placed_with_disk_risk_obs > 0 else None
        )
    }

    placements_df = pd.DataFrame(placements) if dump_placements else None
    return summary, st, rack_util, placements_df


def make_tag(top_pct: float, alpha: float) -> str:
    return f"top{int(top_pct * 100)}_alpha{str(alpha).replace('.', '_')}"


def compute_n_objects_for_target_util(target_util: float) -> int:
    """
    由 target_util 反推 N_OBJECTS，使得：
      cluster_util ~= (N_OBJECTS*REPLICATION*OBJ_SIZE_MB) / total_capacity
    """
    tu = float(target_util)
    total_cap_mb = float(N_RACKS * SLOTS_PER_RACK * SLOT_CAPACITY_MB)
    n_obj = int(round((tu * total_cap_mb) / (REPLICATION * OBJ_SIZE_MB)))
    return max(1, n_obj)


def run_single(out_dir: str, placement_seed: int, target_util: float | None):
    """
    跑一次 step3（一个 seed + 一个 target_util），输出到 out_dir
    """
    os.makedirs(out_dir, exist_ok=True)

    # 复制输入 risk_by_rack 到当前 out_dir，便于 Step4/Step5 就地读取
    if os.path.exists(RISK_BY_RACK):
        rb = pd.read_csv(RISK_BY_RACK)
        rb.to_csv(os.path.join(out_dir, "risk_by_rack.csv"), index=False, encoding="utf-8-sig")

    base_seed = int(placement_seed)

    # load risk_by_rack
    if not os.path.exists(RISK_BY_RACK):
        raise FileNotFoundError(f"missing {RISK_BY_RACK} (run step1 first)")

    risk_by_rack = pd.read_csv(RISK_BY_RACK)
    if not {"rack_id", "risk_score"}.issubset(risk_by_rack.columns):
        raise ValueError("risk_by_rack.csv must have rack_id,risk_score")

    rack_ids, slots_df = make_cluster_topology(N_RACKS, SLOTS_PER_RACK)
    base_state = init_capacity_state(slots_df, SLOT_CAPACITY_MB)

    # --- 根据 target_util 动态设置 N_OBJECTS（全局）---
    global N_OBJECTS
    if target_util is not None:
        if not (0 < float(target_util) <= 1.2):
            raise ValueError("--target_util should be in (0, 1.2]")
        N_OBJECTS = compute_n_objects_for_target_util(float(target_util))
        print(f"[INFO] target_util={float(target_util)} => computed N_OBJECTS={N_OBJECTS}")

    # optional rack-slot risk
    slot_risk_map = None
    if os.path.exists(RACK_SLOT_RISK):
        tmp = pd.read_csv(RACK_SLOT_RISK)
        need = {"rack_id", "slot_id", "risk_disk"}
        if need.issubset(tmp.columns):
            slot_risk_map = tmp[list(need)].copy()
        else:
            print("[WARN] rack_risk_assignment.csv exists but missing columns; ignore it.")

    # load step2 params to know grid
    if not os.path.exists(STEP2_INDEX):
        raise FileNotFoundError(f"missing {STEP2_INDEX} (run step2 first)")
    step2_index = load_json(STEP2_INDEX)
    hard_top_pcts = [float(x) for x in step2_index["hard_top_pcts"]]
    soft_alphas = [float(x) for x in step2_index["soft_alphas"]]

    summaries = []

    for p in hard_top_pcts:
        for a in soft_alphas:
            # 每个参数组合一个独立 RNG，但都锚定在 placement_seed 上
            mix = (int(round(p * 10000)) * 1000003) ^ (int(round(a * 10000)) * 9176) ^ (base_seed * 1315423911)
            setting_seed = abs(mix) % (2**31 - 1)
            rng_setting = np.random.RandomState(setting_seed)

            summary, final_state, rack_util, placements_df = run_one_setting(
                rng_setting, p, a, risk_by_rack, rack_ids, base_state, slot_risk_map,
                dump_placements=DUMP_PLACEMENTS,
                target_util=target_util,
            )

            summary["placement_seed"] = base_seed
            summaries.append(summary)

            tag = make_tag(p, a)

            rack_util.to_csv(os.path.join(out_dir, f"rack_util_{tag}.csv"), index=False, encoding="utf-8-sig")
            final_state.to_csv(os.path.join(out_dir, f"slot_state_{tag}.csv"), index=False, encoding="utf-8-sig")

            if DUMP_PLACEMENTS and placements_df is not None:
                placements_df.to_csv(os.path.join(out_dir, f"placements_{tag}.csv"), index=False, encoding="utf-8-sig")

            print(f"[DONE] {tag} => placed={summary['placed_replicas']} failed_rep={summary['failed_replicas']} util={summary['cluster_utilization']:.4f}")

    summary_df = pd.DataFrame(summaries).sort_values(["top_pct", "alpha"]).reset_index(drop=True)
    out_summary = os.path.join(out_dir, "step3_summary.csv")
    summary_df.to_csv(out_summary, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote {out_summary}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="单次模式输出目录（批量模式会忽略）")
    ap.add_argument("--placement_seed", type=int, default=7, help="单次模式 seed（批量模式会忽略）")
    ap.add_argument("--target_util", type=float, default=None,
                    help="单次模式：目标集群利用率(0~1)。提供则自动计算 N_OBJECTS 逼近该利用率。")

    # 批量模式参数（新增）
    ap.add_argument("--batch_utils", type=str, default=None,
                    help="批量跑多个利用率，例如: 0.05,0.1,0.3,0.5,0.7,0.9,1.0")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4",
                    help="批量模式下 seeds，例如: 0,1,2,3,4")
    ap.add_argument("--root_out", type=str, default=os.path.join("out", "step3_util"),
                    help="批量模式输出根目录，默认 out/step3_util")

    return ap.parse_args()


def main():
    args = parse_args()

    # -------- 批量模式：一键跑 util × seed --------
    if args.batch_utils is not None:
        utils = [float(x.strip()) for x in args.batch_utils.split(",") if x.strip()]
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        root_out = args.root_out

        for tu in utils:
            for sd in seeds:
                out_dir = os.path.join(root_out, f"util_{tu:.2f}", f"seed_{sd}")
                print(f"\n[RUN-BATCH] target_util={tu} seed={sd} => {out_dir}")
                run_single(out_dir=out_dir, placement_seed=sd, target_util=tu)

        print("[DONE] batch step3 finished")
        return

    # -------- 单次模式（保留旧用法）--------
    run_single(out_dir=args.out_dir, placement_seed=args.placement_seed, target_util=args.target_util)


if __name__ == "__main__":
    main()