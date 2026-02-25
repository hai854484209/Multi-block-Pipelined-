# metrics.py
import numpy as np

def sample_client_nodes(cfg, nodes_df, n_reqs, rng):
    """
    为每个请求采样一个“客户端所在节点”，用于估算读请求从哪里发起。
    """
    n_nodes = len(nodes_df)
    rack_id = nodes_df["rack_id"].to_numpy()

    if cfg.client_mode == "uniform":
        # 每次请求的客户端节点在全局节点中均匀采样
        return rng.integers(0, n_nodes, size=n_reqs)

    if cfg.client_mode == "hot_rack":
        # 所有请求都从某个机架发起（机架内节点均匀采样）
        candidates = np.where(rack_id == cfg.hot_rack_id)[0]
        if len(candidates) == 0:
            raise ValueError(f"hot_rack_id={cfg.hot_rack_id} has no nodes.")
        return rng.choice(candidates, size=n_reqs, replace=True)

    if cfg.client_mode == "hot_nodes":
        # 所有请求都从少数热点节点集合发起（热点集合在这里取前 m 个节点）
        if not (0.0 < cfg.hot_node_frac <= 1.0):
            raise ValueError("hot_node_frac must be in (0, 1].")
        m = max(1, int(n_nodes * cfg.hot_node_frac))
        hot = np.arange(m)
        return rng.choice(hot, size=n_reqs, replace=True)

    raise ValueError(f"unknown client_mode: {cfg.client_mode}")


def evaluate_read_metrics(cfg, nodes_df, objects_df, workload_df, placement, bw):
    """
    评估只读场景下的两个核心指标：
    - avg_read_time_s: 平均读时间（近似= size_mb / bandwidth，选带宽最大副本）
    - cross_rack_traffic_mb & cross_rack_ratio: 跨机架读流量与占比
    """
    rng = np.random.default_rng(cfg.seed + 3)
    n_nodes = len(nodes_df)
    rack_id = nodes_df["rack_id"].to_numpy()

    size_mb = objects_df["size_mb"].to_numpy()
    obj_req = workload_df["obj_id"].to_numpy()

    client_nodes = sample_client_nodes(cfg, nodes_df, len(workload_df), rng)

    total_time = 0.0
    cross_rack_mb = 0.0
    total_mb = 0.0

    for i, obj in enumerate(obj_req):
        client = int(client_nodes[i])
        replicas = placement[obj]  # array of node ids, shape (k,)

        # 选择“带宽最大”的副本（等价于时间最小）
        bws = bw[client, replicas]
        best_idx = int(np.argmax(bws))
        chosen = int(replicas[best_idx])

        mb = float(size_mb[obj])
        t = mb / float(bws[best_idx])  # seconds (MB / (MB/s))

        total_time += t
        total_mb += mb

        if rack_id[client] != rack_id[chosen]:
            cross_rack_mb += mb

    return {
        "client_mode": cfg.client_mode,
        "hot_rack_id": cfg.hot_rack_id if cfg.client_mode == "hot_rack" else -1,
        "hot_node_frac": cfg.hot_node_frac if cfg.client_mode == "hot_nodes" else -1,
        "avg_read_time_s": total_time / len(workload_df),
        "cross_rack_traffic_mb": cross_rack_mb,
        "cross_rack_ratio": (cross_rack_mb / total_mb) if total_mb > 0 else 0.0,
    }