import os
import time
import queue
import threading

STUDENT_ID = "230103222"
ITEMS = 1000
WORKERS = 4


def workload(i):
    time.sleep(0.0001 * (i % 20))


def benchmark(strategy):
    task_queue = queue.Queue()

    if strategy == "Dynamic":
        for i in range(ITEMS):
            task_queue.put(i)

    ready = threading.Barrier(WORKERS + 1)
    start_signal = threading.Event()

    finish_times = [0.0] * WORKERS
    processed_counts = [0] * WORKERS

    def worker(worker_id):
        ready.wait()
        start_signal.wait()

        count = 0

        if strategy == "Static":
            start_index = worker_id * ITEMS // WORKERS
            end_index = (worker_id + 1) * ITEMS // WORKERS

            for i in range(start_index, end_index):
                workload(i)
                count += 1

        else:
            while True:
                try:
                    i = task_queue.get_nowait()
                except queue.Empty:
                    break

                workload(i)
                count += 1

        finish_times[worker_id] = time.perf_counter()
        processed_counts[worker_id] = count

    threads = [
        threading.Thread(target=worker, args=(i,))
        for i in range(WORKERS)
    ]

    for thread in threads:
        thread.start()

    # All threads are ready before timing starts.
    ready.wait()
    start = time.perf_counter()
    start_signal.set()

    for thread in threads:
        thread.join()

    end = time.perf_counter()

    if sum(processed_counts) != ITEMS:
        raise RuntimeError("Not all items were processed")

    completion_ms = [
        (finish - start) * 1000
        for finish in finish_times
    ]

    return {
        "wall_ms": (end - start) * 1000,
        "first_ms": min(completion_ms),
        "last_ms": max(completion_ms),
        "delta_ms": max(completion_ms) - min(completion_ms),
        "completion_ms": completion_ms,
        "counts": processed_counts,
    }


def main():
    with open("task4_output.txt", "w", encoding="utf-8") as output:

        def report(text=""):
            print(text, flush=True)
            output.write(text + "\n")
            output.flush()

        report(f"Student ID: {STUDENT_ID}")
        report(f"Logical cores: {os.cpu_count()}")
        report(f"Workers: {WORKERS}")
        report(f"Items: {ITEMS}")
        report("Workload: sleep(0.0001 * (i % 20))")
        report("Dynamic chunk size: 1")
        report("Timing excludes queue setup and thread creation.")
        report()

        results = {}

        for strategy in ("Static", "Dynamic"):
            results[strategy] = benchmark(strategy)

        report(
            f"{'Strategy':<10} | {'Wall ms':>10} | "
            f"{'First ms':>10} | {'Last ms':>10} | "
            f"{'Delta ms':>10}"
        )
        report("-" * 66)

        for strategy, result in results.items():
            report(
                f"{strategy:<10} | "
                f"{result['wall_ms']:>10.3f} | "
                f"{result['first_ms']:>10.3f} | "
                f"{result['last_ms']:>10.3f} | "
                f"{result['delta_ms']:>10.3f}"
            )

        for strategy, result in results.items():
            report()
            report(f"{strategy} worker details:")

            for worker_id in range(WORKERS):
                report(
                    f"Worker {worker_id}: "
                    f"items={result['counts'][worker_id]}, "
                    f"finished={result['completion_ms'][worker_id]:.3f} ms"
                )

        report()
        report(
            "Wall-time ratio (Static / Dynamic): "
            f"{results['Static']['wall_ms'] / results['Dynamic']['wall_ms']:.3f}x"
        )
        report("Verified: 1000 items processed in each strategy.")


if __name__ == "__main__":
    main()