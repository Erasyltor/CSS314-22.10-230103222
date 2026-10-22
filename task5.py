import os
import time
import random
import statistics
import multiprocessing as mp

STUDENT_ID = "230103222"
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
REPEATS = 3

A = 2.5
B = 1.2

X = None
Y = None


def initialize_worker(shared_x, shared_y):
    global X, Y
    X = shared_x
    Y = shared_y


def transform(bounds):
    start, end = bounds


    x = X
    y = Y
    a = A
    b = B

    for i in range(start, end):
        y[i] = a * x[i] + b


def make_chunks(n, workers):
    return [
        (i * n // workers, (i + 1) * n // workers)
        for i in range(workers)
    ]


def measure(action):
    start = time.perf_counter()
    action()
    return (time.perf_counter() - start) * 1000


def verify(n):
    for i in range(n):
        expected = A * X[i] + B
        if Y[i] != expected:
            raise RuntimeError(f"Incorrect result at index {i}")


def main():
    workers = os.cpu_count() or 1
    context = mp.get_context("spawn")

    with open("task5_output.txt", "w", encoding="utf-8") as output:

        def report(text=""):
            print(text, flush=True)
            output.write(text + "\n")
            output.flush()

        report(f"Student ID: {STUDENT_ID}")
        report(f"Workers / logical cores: {workers}")
        report(f"Operation: y[i] = {A} * x[i] + {B}")
        report(f"Repeats: {REPEATS}; reported time: median")
        report("Parallel method: persistent process pool, shared arrays")
        report("Timing includes task dispatch, computation and waiting.")
        report("Allocation, initialization and pool startup are excluded.")
        report()


        shared_x = context.RawArray("d", max(SIZES))
        shared_y = context.RawArray("d", max(SIZES))

        rng = random.Random(STUDENT_ID)
        for i in range(max(SIZES)):
            shared_x[i] = rng.random()

        initialize_worker(shared_x, shared_y)

        results = []

        with context.Pool(
            processes=workers,
            initializer=initialize_worker,
            initargs=(shared_x, shared_y),
        ) as pool:


            startup = time.perf_counter()
            while any(not p.is_alive() for p in pool._pool):
                if time.perf_counter() - startup > 60:
                    raise RuntimeError("Pool startup failed")

            report(
                f"{'N':>12} | {'Serial ms':>12} | "
                f"{'Parallel ms':>12} | {'Speedup':>9} | Result"
            )
            report("-" * 76)

            for n in SIZES:
                chunks = make_chunks(n, workers)

                def serial():
                    transform((0, n))

                def parallel():
                    pool.map(transform, chunks, chunksize=1)


                serial()
                parallel()

                serial_times = []
                parallel_times = []

                for repeat in range(REPEATS):

                    if repeat % 2 == 0:
                        serial_times.append(measure(serial))
                        parallel_times.append(measure(parallel))
                    else:
                        parallel_times.append(measure(parallel))
                        serial_times.append(measure(serial))

                verify(n)

                serial_ms = statistics.median(serial_times)
                parallel_ms = statistics.median(parallel_times)
                speedup = serial_ms / parallel_ms

                results.append((n, serial_ms, parallel_ms))

                report(
                    f"{n:>12,} | {serial_ms:>12.3f} | "
                    f"{parallel_ms:>12.3f} | {speedup:>8.3f}x | "
                    f"{'FASTER' if speedup > 1 else 'SLOWER / EQUAL'}"
                )

        report()
        report("Detected slower-to-faster crossover intervals:")

        crossings = []

        for previous, current in zip(results, results[1:]):
            previous_n, previous_serial, previous_parallel = previous
            current_n, current_serial, current_parallel = current

            if (
                previous_parallel >= previous_serial
                and current_parallel < current_serial
            ):
                crossings.append((previous_n, current_n))
                report(
                    f"N in ({previous_n:,}, {current_n:,}]: "
                    f"slower/equal at {previous_n:,}, "
                    f"faster at {current_n:,}."
                )

        if not crossings:
            report("No slower-to-faster crossover found in tested sizes.")

        faster_sizes = [
            n for n, serial_ms, parallel_ms in results
            if parallel_ms < serial_ms
        ]

        report(
            "Tested sizes where parallel is faster: "
            + (
                ", ".join(f"{n:,}" for n in faster_sizes)
                if faster_sizes else "none"
            )
        )
        report("All computed outputs verified.")


if __name__ == "__main__":
    mp.freeze_support()
    main()