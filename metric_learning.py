# Ilvento's "Metric Learning for Individual Fairness"
# https://arxiv.org/pdf/1906.00250


import numpy as np
from sklearn.preprocessing import normalize


ALPHA = 0.05
NUM_REPS = 10  # Number of representatives.
DATA_PATH = "data/features.csv"
METRIC_DIR = "metric"


def triplet_query(row, x, y):
    """Return whether `x` (0) or `y` (1) is closer to a representative row."""

    return np.argmin([np.linalg.norm(row - x), np.linalg.norm(row - y)])


def real_query(r, x):
    """Return the distance between representative `r` and point `x`."""

    dist = float(input(f"Enter distance between {r} and {x}: "))
    return dist


def merge(row, left, right):
    """Merge two sorted lists with respect to a representative row."""

    res = []
    i = j = 0
    while i < len(left) and j < len(right):
        if triplet_query(row, left[i], right[j]) == 0:
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1
    res.extend(left[i:])
    res.extend(right[j:])
    return res


def merge_sort(row, data):
    """Merge sort with respect to a representative row."""

    if len(data) <= 1:
        return data
    mid = len(data) // 2
    left = merge_sort(row, data[:mid])
    right = merge_sort(row, data[mid:])
    return merge(row, left, right)


def create_submetric(r, data):
    """Create an alpha-submetric with respect to representative `r`."""

    data = merge_sort(data[r], data)
    D = np.zeros((len(data), len(data)))

    def label(left, right):
        if abs(real_query(r, left) - real_query(r, right)) > ALPHA:
            mid = (left + right) // 2
            label(left, mid)
            label(mid + 1, right)
        else:
            for x in range(left, right + 1):
                D[r][x] = real_query(r, left)

    label(0, len(data) - 1)

    for i in range(len(data)):
        for j in range(len(data)):
            D[i][j] = abs(D[r][i] - D[r][j])

    return D


def max_merge(num_reps, data):
    """Max merge `num_reps` number of different alpha-submetrics."""

    reps = np.random.choice(len(data), num_reps, replace=False)
    D = np.zeros((len(data), len(data)))
    for r in reps:
        D = np.maximum(D, create_submetric(r, data))
    return D


def main():
    """Find an alpha-submetric."""

    with open(DATA_PATH, "rb") as f:
        data = np.loadtxt(f, delimiter=",", skiprows=1, dtype=np.float32)

    data = normalize(data, axis=0)
    D = max_merge(NUM_REPS, data)
    print(D)


if __name__ == "__main__":
    main()
