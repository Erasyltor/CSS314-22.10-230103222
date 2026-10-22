import sys
import time
import threading

STUDENT_ID = "230103222"
WORKERS = 4
ITERATIONS = 2_000_000
EXPECTED = WORKERS * ITERATIONS

counter = 0


def next_value(value):
    return value + 1


def increment():
    global counter

    current = counter
    updated = next_value(current)
    counter = updated


def benchmark(use_lock):
    global counter
    counter = 0

    lock = threading.Lock()
    ready = threading.Barrier(WORKERS + 1)
    start_signal = threading.Event()

    def worker():
        ready.wait()
        start_signal.wait()

        if use_lock:
            for _ in range(ITERATIONS):
                with lock:
                    increment()
        else:
            for _ in range(ITERATIONS):
                increment()

    threads = [
        threading.Thread(target=worker)
        for _ in range(WORKERS)
    ]

    for thread in threads:
        thread.start()

    ready.wait()

    start = time.perf_counter()
    start_signal.set()

    for thread in threads:
        thread.join()

    elapsed_ms = (time.perf_counter() - start) * 1000
    return counter, elapsed_ms


def main():
    original_interval = sys.getswitchinterval()
    sys.setswitchinterval(0.0001)

    try:
        with open("task6_output.txt", "w", encoding="utf-8") as output:

            def report(text=""):
                print(text, flush=True)
                output.write(text + "\n")
                output.flush()

            report(f"Student ID: {STUDENT_ID}")
            report(f"Python: {sys.version.split()[0]}")
            report(f"Threads: {WORKERS}")
            report(f"Increments per thread: {ITERATIONS:,}")
            report(f"Expected final value: {EXPECTED:,}")
            report(
                f"Thread switch interval: "
                f"{sys.getswitchinterval():.6f} seconds"
            )
            report("Timing excludes thread creation.")
            report()

            report("Running WITHOUT lock...")
            unlocked_value, unlocked_ms = benchmark(False)

            report("Running WITH lock...")
            locked_value, locked_ms = benchmark(True)

            unlocked_lost = EXPECTED - unlocked_value
            locked_lost = EXPECTED - locked_value

            report()
            report(
                f"{'Version':<14} | {'Actual value':>12} | "
                f"{'Lost updates':>12} | {'Lost %':>9} | "
                f"{'Wall ms':>12}"
            )
            report("-" * 74)

            report(
                f"{'Without lock':<14} | {unlocked_value:>12,} | "
                f"{unlocked_lost:>12,} | "
                f"{100 * unlocked_lost / EXPECTED:>8.3f}% | "
                f"{unlocked_ms:>12.3f}"
            )

            report(
                f"{'With lock':<14} | {locked_value:>12,} | "
                f"{locked_lost:>12,} | "
                f"{100 * locked_lost / EXPECTED:>8.3f}% | "
                f"{locked_ms:>12.3f}"
            )

            report()
            report(
                f"Synchronization time penalty: "
                f"{locked_ms - unlocked_ms:.3f} ms"
            )
            report(
                f"Locked / unlocked time ratio: "
                f"{locked_ms / unlocked_ms:.3f}x"
            )
            report(
                f"Time change: "
                f"{(locked_ms / unlocked_ms - 1) * 100:.2f}%"
            )

            if locked_value != EXPECTED:
                raise RuntimeError("Locked counter is incorrect")

            report("Verified: locked counter equals 8,000,000.")

            if unlocked_lost == 0:
                report(
                    "No race manifested in this run. "
                    "This does not prove the unlocked version is safe."
                )

    finally:
        sys.setswitchinterval(original_interval)


if __name__ == "__main__":
    main()