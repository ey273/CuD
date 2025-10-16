from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class SCFModel:
    xor_steps: int
    count_steps: int
    t_ap: float
    parallelization: int

    def time_per_query(self, n: int, d: int) -> float:
        xor_time = d * self.xor_steps * self.t_ap
        count_time = d * self.count_steps * self.t_ap
        return (n / max(self.parallelization, 1)) * (xor_time + count_time)


@dataclass
class MemoryModel:
    bandwidth_GBps: float
    link_latency_ns: float

    def _bandwidth_Bps(self) -> float:
        # Convert GB/s → B/s
        return self.bandwidth_GBps * 1e9

    def transfer_time(self, bytes_to_move: int) -> float:
        Bps = max(self._bandwidth_Bps(), 1.0)
        data_time = bytes_to_move / Bps
        fixed = self.link_latency_ns * 1e-9
        return data_time + fixed


@dataclass
class RerankModel:
    flop_rate_TFLOPs: float

    def dot_time(self, m: int, d: int) -> float:
        flops = 2.0 * m * d 
        tflops = max(self.flop_rate_TFLOPs, 1e-9) * 1e12
        return flops / tflops


@dataclass
class PipelineConfig:
    n: int
    d: int
    nq: int
    k: int
    filter_ratio: float
    scf: SCFModel
    memory: MemoryModel
    rerank: RerankModel
    bits_per_item: int
    elem_bytes: int
    init_multiplier: float       
    scf_multiplier: float
    xfer_multiplier: float
    rerank_multiplier: float

    def bitmap_size_bytes(self) -> int:
        total_bits = self.n * self.bits_per_item
        return (total_bits + 7) // 8

    def m_candidates(self) -> int:
        return max(1, int(self.n * self.filter_ratio))


def memory_presets() -> Dict[str, MemoryModel]:
    return {
        "HBM": MemoryModel(bandwidth_GBps=1200.0, link_latency_ns=100.0),
        "DDR": MemoryModel(bandwidth_GBps=280.0, link_latency_ns=100.0),
        "CXL": MemoryModel(bandwidth_GBps=32.0,  link_latency_ns=250.0),
    }


def xpu_presets() -> Dict[str, RerankModel]:
    return {
        "GPU": RerankModel(flop_rate_TFLOPs=900.0),
        "CPU": RerankModel(flop_rate_TFLOPs=3.0),
    }


def simulate_once(cfg: PipelineConfig) -> Dict[str, float]:
    query_bytes = cfg.d * cfg.elem_bytes
    t_init = cfg.init_multiplier * cfg.memory.transfer_time(query_bytes)

    m = cfg.m_candidates()
    bitmap_bytes = cfg.bitmap_size_bytes()

    t_scf = cfg.scf_multiplier * cfg.scf.time_per_query(cfg.n, cfg.d)
    t_bitmap = cfg.xfer_multiplier * cfg.memory.transfer_time(bitmap_bytes) 

    t_rerank_compute = cfg.rerank_multiplier * cfg.rerank.dot_time(m, cfg.d)
    t_rerank_mem = 2*(cfg.xfer_multiplier * cfg.memory.transfer_time(m * cfg.d * cfg.elem_bytes)) / 0.5
    t_rerank = max(t_rerank_compute, t_rerank_mem)

    total_cud = t_scf + t_bitmap + t_rerank
    return {
        "t_init": t_init,  
        "t_scf": t_scf,
        "t_bitmap": t_bitmap,
        "t_rerank": t_rerank,
        "total_cud": total_cud,
        "t_rerank_compute": t_rerank_compute,
        "t_rerank_mem": t_rerank_mem,
    }


def run_simulation(cfg: PipelineConfig) -> Tuple[Dict[str, float], Dict[str, float]]:
    pq = simulate_once(cfg)
    nq = cfg.nq
    
    total_cud = pq["total_cud"] * nq
    total_init = pq["t_init"] * nq 
    total_scf = pq["t_scf"] * nq
    total_xfer = pq["t_bitmap"] * nq
    total_rer = pq["t_rerank"] * nq

    eps = 1e-18
    qps_cud = nq / max(total_cud, eps)
    init_pct = 100.0 * total_init / max(total_cud, eps)
    scf_pct = 100.0 * total_scf / max(total_cud, eps)
    bitmap_pct = 100.0 * total_xfer / max(total_cud, eps)
    rerank_pct = 100.0 * total_rer / max(total_cud, eps)

    aggregated = {
        "nq": nq,
        "total_time_cud_s": total_cud,
        "qps_cud": qps_cud,
        "init_pct_of_cud": init_pct,  
        "scf_pct_of_cud": scf_pct,
        "bitmap_pct_of_cud": bitmap_pct,
        "rerank_pct_of_cud": rerank_pct,
    }
    return pq, aggregated


if __name__ == "__main__":
    mem = memory_presets()["DDR"]
    xpu = xpu_presets()["CPU"]
    scf = SCFModel(
        xor_steps=7,
        count_steps=7,
        t_ap=50e-9,
        parallelization=8192 * 32 * 4 * 8 * 8
    )
    cfg = PipelineConfig(
        n=9_990_000,
        d=96,
        nq=1000,
        k=32,
        filter_ratio=1/100,
        scf=scf,
        memory=mem,
        rerank=xpu,
        bits_per_item=1,
        elem_bytes=4,
        init_multiplier=1.0,      
        scf_multiplier=1.0,
        xfer_multiplier=1.0,
        rerank_multiplier=1.0,
    )
    per_query, agg = run_simulation(cfg)
    for k, v in agg.items():
        if "qps" in k:
            print(f"{k:>18}: {v:.2f}")
        else:
            print(f"{k:>18}: {v:.6f}")
