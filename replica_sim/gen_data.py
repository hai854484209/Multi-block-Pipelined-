# gen_data.py
import numpy as np
import pandas as pd

def gen_nodes(cfg):
    rng = np.random.default_rng(cfg.seed)
    n_nodes = cfg.n_racks * cfg.nodes_per_rack
    node_ids = np.arange(n_nodes)
    rack_id = node_ids // cfg.nodes_per_rack
    bg_load = rng.uniform(cfg.bg_load_min, cfg.bg_load_max, size=n_nodes)
    return pd.DataFrame({"node_id": node_ids, "rack_id": rack_id, "bg_load": bg_load})

def gen_objects(cfg):
    obj_ids = np.arange(cfg.n_objects)
    return pd.DataFrame({"obj_id": obj_ids, "size_mb": cfg.obj_size_mb})

def gen_workload(cfg):
    rng = np.random.default_rng(cfg.seed + 1)
    # Zipf over object ids
    ranks = np.arange(1, cfg.n_objects + 1)
    weights = 1.0 / (ranks ** cfg.zipf_s)
    probs = weights / weights.sum()
    obj_ids = rng.choice(cfg.n_objects, size=cfg.n_requests, p=probs)
    return pd.DataFrame({"t": np.arange(cfg.n_requests), "obj_id": obj_ids})

def gen_bandwidth_matrix(cfg):
    n_nodes = cfg.n_racks * cfg.nodes_per_rack
    bw = np.full((n_nodes, n_nodes), cfg.bw_inter_rack, dtype=float)
    for i in range(n_nodes):
        bw[i, i] = np.inf
    # same rack -> intra rack bandwidth
    for r in range(cfg.n_racks):
        start = r * cfg.nodes_per_rack
        end = start + cfg.nodes_per_rack
        bw[start:end, start:end] = cfg.bw_intra_rack
        np.fill_diagonal(bw[start:end, start:end], np.inf)
    return bw