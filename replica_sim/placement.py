# placement.py
import numpy as np


def random_placement(cfg, nodes_df):
    """
    标准随机放置基线：
    - 对每个对象，先随机选择 k 个不同的 rack（不放回抽样）
    - 再在每个 rack 内随机选择 1 个节点作为副本落点
    这样可保证：k=3 时副本必然跨 3 个 rack，rack 维度更均衡、也更符合常见“故障域隔离”基线。
    """
    rng = np.random.default_rng(cfg.seed + 2)

    n_nodes = len(nodes_df)
    rack_id = nodes_df["rack_id"].to_numpy()
    racks = np.unique(rack_id)
    n_racks = len(racks)

    k = cfg.replica_k
    if k > n_racks:
        raise ValueError(f"replica_k={k} cannot exceed n_racks={n_racks} when enforcing all-different racks.")

    # 预先按 rack 收集节点列表，便于快速采样
    nodes_in_rack = {}
    for r in racks:
        nodes_in_rack[int(r)] = np.where(rack_id == r)[0]

    placement = np.empty((cfg.n_objects, k), dtype=int)

    for obj in range(cfg.n_objects):
        chosen_racks = rng.choice(racks, size=k, replace=False)  # k 个不同 rack
        rep_nodes = []
        for r in chosen_racks:
            candidates = nodes_in_rack[int(r)]
            rep_nodes.append(int(rng.choice(candidates)))
        placement[obj] = rep_nodes

    return placement