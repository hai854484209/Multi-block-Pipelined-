# config.py
from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 7

    # cluster
    n_racks: int = 4
    nodes_per_rack: int = 16   # 4*16 = 64
    bg_load_min: float = 0.1
    bg_load_max: float = 0.9

    # objects
    n_objects: int = 2000
    replica_k: int = 3
    obj_size_mb: int = 64

    # workload
    n_requests: int = 20000
    zipf_s: float = 1.1   # Zipf 参数，1.0~1.3 常用

    # network (MB/s)
    bw_intra_rack: float = 800.0
    bw_inter_rack: float = 120.0

    # client request source model
    # "uniform": 客户端节点全局均匀随机
    # "hot_rack": 客户端集中在某个机架内的节点
    # "hot_nodes": 客户端集中在少数热点节点集合
    client_mode: str = "uniform"
    hot_rack_id: int = 0
    hot_node_frac: float = 0.1  # hot_nodes 时热点节点占比（0~1）