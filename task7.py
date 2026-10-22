import os
import math
import time
import platform
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

STUDENT_ID = "230103222"
SIZE = 30_000_000
REPEATS = 5


@njit(nogil=True)
def memory_kernel(a, b, start, end):
    for i in range(start, end):
        b[i] = a[i]


@njit(nogil=True)
def compute_kernel(a, b, start, end):
    for i in range(start, end):
        value = a[i]
        b[i] = (
            math.sin(value) * math.cos(value)
            + math.tan(value)
        )


def benchmark(kernel, a, b, workers):
    chunks = [
        (i * SIZE // workers, (i + 1) * SIZE // workers)
        for i in range(workers)
    ]

    with ThreadPoolExecutor(max_workers=workers) as pool:
      
        ready = threading.Barrier(workers + 1)

        def wait_until_ready():
            ready.wait()

        startup = [
            pool.submit(wait_until_ready)
            for _ in range(workers)
        ]

        ready.wait()

        for future in startup:
            future.result()

        def execute():
            futures = [
                pool.submit(kernel, a, b, start, end)
                for start, end in chunks
            ]
            for future in futures:
                future.result()

        execute()

        times = []

        for _ in range(REPEATS):
            start = time.perf_counter()
            execute()
            times.append(time.perf_counter() - start)

    return statistics.median(times)


def verify(name, a, b):
   
    indices = np.linspace(0, SIZE - 1, 10_000, dtype=np.int64)
    values = a[indices]

    if name == "A: Copy":
        np.testing.assert_array_equal(b[indices], values)
    else:
        expected = np.sin(values) * np.cos(values) + np.tan(values)
        np.testing.assert_allclose(
            b[indices], expected, rtol=1e-12, atol=1e-12
        )


def main():
    max_cores = os.cpu_count() or 1
    worker_counts = sorted(set([1, 2, max_cores]))

    with open("task7_output.txt", "w", encoding="utf-8") as output:

        def report(text=""):
            print(text, flush=True)
            output.write(text + "\n")
            output.flush()

        report(f"Student ID: {STUDENT_ID}")
        report(f"OS: {platform.platform()}")
        report(f"CPU: {platform.processor() or platform.machine()}")
        report(f"Available logical cores: {max_cores}")
        report(f"Elements per array: {SIZE:,}")
        report("Data type: float64")
        report(f"Repeats: {REPEATS}; reported time: median")
        report("Timing includes task submission and waiting.")
        report("Compilation, allocation and thread startup are excluded.")
        report()

       
        rng = np.random.default_rng(int(STUDENT_ID))
        a = rng.random(SIZE)
        b = np.empty_like(a)

       
        memory_kernel(a, b, 0, 1)
        compute_kernel(a, b, 0, 1)

        results = {}

        report(
            f"{'Kernel':<12} | {'Cores':>5} | {'Wall ms':>11} | "
            f"{'Speedup':>9} | {'Efficiency':>10} | {'GB/s*':>9}"
        )
        report("-" * 78)

        for name, kernel in [
            ("A: Copy", memory_kernel),
            ("B: Compute", compute_kernel),
        ]:
            results[name] = {}
            baseline = None

            for workers in worker_counts:
                elapsed = benchmark(kernel, a, b, workers)
                verify(name, a, b)

                if baseline is None:
                    baseline = elapsed

                speedup = baseline / elapsed
                efficiency = speedup / workers

        
                bandwidth = (SIZE * 8 * 2) / elapsed / 1e9

                results[name][workers] = elapsed

                report(
                    f"{name:<12} | {workers:>5} | "
                    f"{elapsed * 1000:>11.3f} | "
                    f"{speedup:>8.3f}x | "
                    f"{efficiency:>9.2%} | "
                    f"{bandwidth:>9.3f}"
                )

        report()
        report("* GB/s is useful array traffic, not measured RAM-bus traffic.")
        report("Sampled output checks passed for every run configuration.")
        report()
        report("Scaling assessment:")

        if max_cores > 2:
            copy_gain = (
                results["A: Copy"][2]
                / results["A: Copy"][max_cores]
            )
            compute_gain = (
                results["B: Compute"][2]
                / results["B: Compute"][max_cores]
            )

            report(f"Copy speed gain from 2 to max cores: {copy_gain:.3f}x")
            report(f"Compute speed gain from 2 to max cores: {compute_gain:.3f}x")

            
            if copy_gain <= 1.20:
                report(
                    "Kernel A (Copy) stalled in scaling "
                    "(<=20% improvement from 2 to max cores)."
                )
                report(
                    "This is consistent with memory-channel saturation, "
                    "but timing alone cannot prove the cause."
                )
            else:
                report(
                    "Kernel A (Copy) did not meet the scaling-stall "
                    "criterion; RAM saturation was not established."
                )

            if compute_gain <= 1.20:
                report(
                    "Kernel B also stalled. Scheduling, CPU limits "
                    "or other bottlenecks may affect these results."
                )
            else:
                report(
                    "Kernel B continued to benefit from additional workers."
                )
        else:
            report(
                "More than 2 available cores are needed to assess "
                "a plateau between 2 and max cores."
            )


if __name__ == "__main__":
    main()