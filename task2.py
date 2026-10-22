import os
import platform
import random
import time
import math
import multiprocessing as mp
from multiprocessing import shared_memory
from array import array

STUDENT_ID = "230103222"
SIZE = 20_000_000
P = 0.95

worker_memory = None
worker_data = None


def initialize_worker(memory_name):
    global worker_memory, worker_data

    worker_memory = shared_memory.SharedMemory(name=memory_name)
    worker_data = worker_memory.buf.cast("d")


def sum_chunk(bounds):
    start, end = bounds
    return sum(worker_data[start:end])


def benchmark(worker_count, memory_name):

    chunks = [
        (i * SIZE // worker_count, (i + 1) * SIZE // worker_count)
        for i in range(worker_count)
    ]

    start = time.perf_counter()

    with mp.Pool(
        processes=worker_count,
        initializer=initialize_worker,
        initargs=(memory_name,),
    ) as pool:
        partial_sums = pool.map(sum_chunk, chunks)
        total = math.fsum(partial_sums)

    elapsed = time.perf_counter() - start
    return total, elapsed


def main():
    with open("task2_output.txt", "w", encoding="utf-8") as output_file:

        def report(text=""):
            print(text, flush=True)
            output_file.write(text + "\n")
            output_file.flush()

        max_cores = os.cpu_count() or 1
        worker_counts = list(
            dict.fromkeys([1, 2, 4, max_cores, max_cores * 2])
        )

        report(f"Student ID: {STUDENT_ID}")
        report(f"OS: {platform.platform()}")
        report(
            f"CPU / platform: "
            f"{platform.processor() or platform.machine()}"
        )
        report(f"Logical cores: {max_cores}")
        report(f"Elements: {SIZE}")
        report(f"Parallelizable fraction: {P}")
        report("Timing includes process startup, summation and shutdown.")
        report("Array generation and shared-memory setup are excluded.")
        report()

        rng = random.Random(STUDENT_ID)
        data = array("d", (rng.random() for _ in range(SIZE)))

        memory = shared_memory.SharedMemory(
            create=True,
            size=len(data) * data.itemsize,
        )

        try:
           
            source = memoryview(data).cast("B")
            try:
                memory.buf[:] = source
            finally:
                source.release()

            del data

            report(
                f"{'Cores':>7} | {'Wall Time (ms)':>16} | "
                f"{'Speedup':>10} | {'Amdahl Upper Bound':>19}"
            )
            report("-" * 63)

            baseline_time = None
            reference_sum = None

            for workers in worker_counts:
                total, elapsed = benchmark(workers, memory.name)

                if baseline_time is None:
                    baseline_time = elapsed
                    reference_sum = total

                speedup = baseline_time / elapsed
                amdahl = 1 / ((1 - P) + P / workers)

                report(
                    f"{workers:>7} | {elapsed * 1000:>16.3f} | "
                    f"{speedup:>9.3f}x | {amdahl:>18.3f}x"
                )

                if not math.isclose(
                    total, reference_sum, rel_tol=1e-10, abs_tol=1e-8
                ):
                    raise RuntimeError(
                        f"Sum mismatch for {workers} workers: {total}"
                    )

            report()
            report(f"Sum (1 worker): {reference_sum:.12f}")
            report("All parallel sums match within floating-point tolerance.")

        finally:
            memory.close()
            memory.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()
