

import torch
import time
from deep_gemm import gemm_fp8_fp8_bf16_nt  # Import function from gemm.py


def benchmark_gemm(m, n, k, num_runs=50):
    device = 'cuda'

    # Create random input tensors
    A = torch.randn(m, k, dtype=torch.float8_e4m3fn, device=device)
    B = torch.randn(n, k, dtype=torch.float8_e4m3fn, device=device)
    C = torch.zeros(m, n, dtype=torch.bfloat16, device=device)

    lhs = (A, torch.randn((m, (k + 127) // 128), dtype=torch.float32, device=device))
    rhs = (B, torch.randn(((n + 127) // 128, (k + 127) // 128), dtype=torch.float32, device=device))

    # Run benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        gemm_fp8_fp8_bf16_nt(lhs, rhs, C)
    torch.cuda.synchronize()
    elapsed_time = (time.time() - start) / num_runs

    print(f"GEMM execution time: {elapsed_time * 1e6:.2f} µs")


if __name__ == "__main__":
    benchmark_gemm(m=2048, n=2048, k=1024)

