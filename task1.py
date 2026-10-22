import os
import platform
import random
import time
import math
from array import array

STUDENT_ID = "230103222"  # Впиши свой Student ID

SIZE = 20_000_000


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU / platform: {platform.processor() or platform.machine()}")
    print(f"Logical cores: {os.cpu_count()}")
    print(f"Elements: {SIZE}")
    print("Execution: serial, single thread")

    total_start = time.perf_counter()

    # Генерация массива float64 с seed на основе Student ID.
    rng = random.Random(STUDENT_ID)
    data = array("d", (rng.random() for _ in range(SIZE)))

    calculation_start = time.perf_counter()

    # Сумма и стандартное отклонение за один последовательный проход.
    # Алгоритм Уэлфорда для устойчивого вычисления дисперсии.
    total = 0.0
    mean = 0.0
    m2 = 0.0

    for count, value in enumerate(data, start=1):
        total += value
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)

    # Стандартное отклонение всей совокупности: делим на N.
    std_dev = math.sqrt(m2 / SIZE)

    end = time.perf_counter()

    print(f"Sum: {total:.12f}")
    print(f"Standard deviation (population): {std_dev:.12f}")
    print(f"Generation time: {(calculation_start - total_start) * 1000:.3f} ms")
    print(f"Calculation time: {(end - calculation_start) * 1000:.3f} ms")
    print(f"Total wall-clock time: {(end - total_start) * 1000:.3f} ms")


if __name__ == "__main__":
    main()