import os
os.environ["OMP_NUM_THREADS"] = "36"
os.environ["OPENBLAS_NUM_THREADS"] = "36"
import time
import numpy as np


def initialize_data(n: int, d: int, nq: int, seed: int):
    rng = np.random.default_rng(seed)
    xb = rng.standard_normal((n, d), dtype=np.float32)
    xq = rng.standard_normal((nq, d), dtype=np.float32)
    return xb, xq


def scf(n: int, d: int):
    xor_steps        = 7
    count_steps      = 7
    t_AP             = 50e-9
    parallelization  = 8192*32*4*8*8
    xor_time   = d * xor_steps   * t_AP
    count_time = d * count_steps * t_AP
    return (n / parallelization) * (xor_time + count_time)


def bitmap(bits: float, bandwidth_GBps: float) -> float:
    if bandwidth_GBps is None or bandwidth_GBps <= 0:
        return 0.0
    return bits / (bandwidth_GBps * 8e9)


def reranking_breakdown(xb_subset: np.ndarray, q_i: np.ndarray, k: int):
    t0 = time.perf_counter()
    scores = np.dot(xb_subset, q_i)
    t_dot = time.perf_counter() - t0

    k = min(k, scores.shape[0])
    t1 = time.perf_counter()
    topk_idx = np.argpartition(scores, -k)[-k:]
    topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]
    t_topk = time.perf_counter() - t1

    return max(t_dot, 1e-12), max(t_topk, 1e-12), max(t_dot + t_topk, 1e-12)


def main():
    # params
    n = 9_990_000
    d = 96
    filter_ratio = 1 / 100
    nq = 1000
    k = 32
    seed = 42
    bandwidth_GBps = 270.0

    rng = np.random.default_rng(seed)
    xb, xq = initialize_data(n, d, nq, seed)

    total_scf_time = 0.0
    total_bitmap_time = 0.0
    total_dot_time = 0.0
    total_topk_time = 0.0

    m = max(1, int(n * filter_ratio))

    for i in range(nq):
        q_i = xq[i]

        total_scf_time += scf(n, d)
        total_bitmap_time += bitmap(n, bandwidth_GBps)

        cand_idx = rng.choice(n, size=m, replace=False)
        xb_subset = xb[cand_idx]

        t_dot, t_topk, _ = reranking_breakdown(xb_subset, q_i, k)
        total_dot_time += t_dot
        total_topk_time += t_topk

    total_reranking_time = total_dot_time + total_topk_time

    total_time_cud = total_scf_time + total_bitmap_time + total_reranking_time
    total_time_drex = total_bitmap_time + total_reranking_time

    qps_cud = nq / max(total_time_cud, 1e-12)
    qps_drex = nq / max(total_time_drex, 1e-12)

    print(f"CuD QPS: {qps_cud:.2f}")
    print(f"DReX QPS: {qps_drex:.2f}")

    denom = max(total_time_cud, 1e-12)
    print(f"SCF: %{(100 * total_scf_time / denom):.2f}")
    print(f"Bitmap: %{(100 * total_bitmap_time / denom):.2f}")
    print(f"Dot (xb_subset @ q_i): %{(100 * total_dot_time / denom):.2f}")
    print(f"Top-k (argpartition/sort): %{(100 * total_topk_time / denom):.2f}")

    print(f"Reranking total (dot + top-k): %{(100 * total_reranking_time / denom):.2f}")


if __name__ == "__main__":
    main()
