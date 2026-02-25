# main.py
import os
import pandas as pd
import numpy as np

from config import Config
from gen_data import gen_nodes, gen_objects, gen_workload, gen_bandwidth_matrix
from placement import random_placement
from metrics import evaluate_read_metrics

# 验证函数
def verify_placement_distinct_racks(nodes_df, placement, k_expected, n_check=200):
    rack_id = nodes_df["rack_id"].to_numpy()
    n_objs = placement.shape[0]
    n = min(n_check, n_objs)

    distinct = np.array([len(set(rack_id[placement[o]])) for o in range(n)])
    print(f"[VERIFY] distinct racks per object (first {n} objs): "
          f"min={distinct.min()}, max={distinct.max()}, mean={distinct.mean():.2f}")

    # 强制检查：必须每个对象都是 k 个不同 rack
    if distinct.min() != k_expected or distinct.max() != k_expected:
        bad = np.where(distinct != k_expected)[0][:10]
        raise AssertionError(f"Placement violates all-different-rack constraint. "
                             f"Example bad obj ids (first 10): {bad.tolist()}")

def run_one(cfg: Config):
    nodes = gen_nodes(cfg)
    objects = gen_objects(cfg)
    workload = gen_workload(cfg)
    bw = gen_bandwidth_matrix(cfg)

    placement = random_placement(cfg, nodes)

    # === 验证：每个对象的副本是否落在 k 个不同 rack ===
    verify_placement_distinct_racks(nodes, placement, k_expected=cfg.replica_k, n_check=200)

    m = evaluate_read_metrics(cfg, nodes, objects, workload, placement, bw)
    return m, nodes, objects, workload


def main():
    os.makedirs("out", exist_ok=True)

    base_cfg = Config()

    results = []

    # 1) uniform
    cfg = Config(**vars(base_cfg))
    cfg.client_mode = "uniform"
    m, nodes, objects, workload = run_one(cfg)
    results.append(m)

    # 2) hot_rack
    cfg = Config(**vars(base_cfg))
    cfg.client_mode = "hot_rack"
    cfg.hot_rack_id = 0
    m, _, _, _ = run_one(cfg)
    results.append(m)

    # 3) hot_nodes
    cfg = Config(**vars(base_cfg))
    cfg.client_mode = "hot_nodes"
    cfg.hot_node_frac = 0.1
    m, _, _, _ = run_one(cfg)
    results.append(m)

    df = pd.DataFrame(results)
    df.to_csv("out/metrics_by_client_mode.csv", index=False)

    # 保存一次输入数据（用 uniform 那次的，保证可复现）
    nodes.to_csv("out/nodes.csv", index=False)
    objects.to_csv("out/objects.csv", index=False)
    workload.to_csv("out/workload.csv", index=False)

    print(df)


if __name__ == "__main__":
    main()