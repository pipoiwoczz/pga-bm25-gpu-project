import subprocess
import sys
import os

import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
HERE = os.path.dirname(os.path.abspath(__file__))

BENCHES = [
    "bench_cpu_vs_library.py",
    "bench_gpu_v1.py",
    "bench_gpu_v2.py",
]


def main():
    extra = sys.argv[1:]  # forward all CLI flags to each bench
    for bench in BENCHES:
        print("\n" + "=" * 60)
        print(f"  {bench}")
        print("=" * 60)
        ret = subprocess.run(
            [sys.executable, os.path.join(HERE, bench), *extra])
        if ret.returncode != 0:
            print(f"  [warn] {bench} exited with code {ret.returncode} "
                  f"(GPU benches exit non-zero without CUDA — expected on CPU-only machines)")


if __name__ == "__main__":
    main()