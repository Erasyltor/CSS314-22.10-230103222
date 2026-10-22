import os
import time
import ctypes
import platform
import subprocess
import multiprocessing as mp

STUDENT_ID = "230103222"
ITERATIONS = 50_000_000

# 8 элементов по 8 байт = расстояние 64 байта.
PADDED_STRIDE = 8


def physical_core_count():
    if platform.system() == "Windows":
        command = (
            "(Get-CimInstance Win32_Processor | "
            "Measure-Object -Property NumberOfCores -Sum).Sum"
        )

        result = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command],
            text=True,
        )

        return int(result.strip())

    # На других ОС количество физических ядер вводится вручную.
    return int(input("Enter physical core count: "))


def worker(shared_counts, index, ready, start_signal):
    ready.release()
    start_signal.wait()

    for _ in range(ITERATIONS):
        shared_counts[index] += 1


def benchmark(workers, stride):
    shared_counts = mp.RawArray(
        ctypes.c_longlong, workers * stride
    )

    ready = mp.Semaphore(0)
    start_signal = mp.Event()

    processes = [
        mp.Process(
            target=worker,
            args=(
                shared_counts,
                i * stride,
                ready,
                start_signal,
            ),
        )
        for i in range(workers)
    ]

    for process in processes:
        process.start()

    # Ждём готовности всех процессов перед измерением.
    for _ in processes:
        ready.acquire()

    start = time.perf_counter()
    start_signal.set()

    for process in processes:
        process.join()

    elapsed_ms = (time.perf_counter() - start) * 1000

    for process in processes:
        if process.exitcode != 0:
            raise RuntimeError("Worker process failed")

    for i in range(workers):
        if shared_counts[i * stride] != ITERATIONS:
            raise RuntimeError("Incorrect counter value")

    return elapsed_ms


def main():
    workers = physical_core_count()

    if workers < 1:
        raise ValueError("Physical core count must be positive")

    with open("task3_output.txt", "w", encoding="utf-8") as output:

        def report(text=""):
            print(text, flush=True)
            output.write(text + "\n")
            output.flush()

        report(f"Student ID: {STUDENT_ID}")
        report(f"OS: {platform.platform()}")
        report(f"Logical cores: {os.cpu_count()}")
        report(f"Physical cores / workers: {workers}")
        report("Worker type: processes")
        report(f"Increments per worker: {ITERATIONS}")
        report("Padded counter spacing: 64 bytes")
        report("Timing includes worker release, work and joining.")
        report("Process startup is excluded.")
        report()

        report("Running Unpadded Shared...")
        shared_ms = benchmark(workers, 1)

        report("Running Padded...")
        padded_ms = benchmark(workers, PADDED_STRIDE)

        report()
        report(f"{'Version':<22} | {'Wall Time (ms)':>16}")
        report("-" * 43)
        report(f"{'Unpadded Shared':<22} | {shared_ms:>16.3f}")
        report(f"{'Padded':<22} | {padded_ms:>16.3f}")
        report()
        report(
            f"Execution ratio (Shared / Padded): "
            f"{shared_ms / padded_ms:.3f}x"
        )
        report("All counter values verified.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
    