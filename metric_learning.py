# Ilvento's "Metric Learning for Individual Fairness"
# https://arxiv.org/pdf/1906.00250


import numpy as np
import os
from sklearn.preprocessing import normalize
from tqdm import tqdm


ALPHA = 0.1
NUM_REPS = 5  # Number of representatives.
DATA_DIR = "data"
METRIC_DIR = "metric"


submetrics = {}  # submetrics.get(r)[x][y] = distance between `x` and `y` w.r.t. `r`.
real_dists = []  # real_dists[r][x] = distance between `r` and `x`.

bounds = {}  # bounds[x] = (dist. on left, dist. on right) of `x`.


def triplet_query(row, x, y):
    """Return whether `x` (0) or `y` (1) is closer to a representative row."""

    return np.argmin([np.linalg.norm(row[1:] - x[1:]), np.linalg.norm(row[1:] - y[1:])])


def real_query(r, x):
    """Return the distance between representative `r` and point `x`."""

    if real_dists[r][x] != -1.0:
        return real_dists[r][x]

    with open(f"{DATA_DIR}/preonehot.csv", "rb") as f:
        data = np.loadtxt(f, delimiter=",", dtype=np.dtypes.StrDType)
        cols, row_r, row_x = data[0], data[r + 1], data[x + 1]
        print()
        for a, b, c in zip(cols, row_r, row_x):
            print(f"{a}: {b} vs. {c}")
        bnds = bounds.get(x, (-1, -1))
        print(f"bounds: {bnds[0]} vs. {bnds[1]}")

    dist = float(input(f"Enter distance between {r} and {x}: "))
    real_dists[r][x] = dist
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
    return np.array(res)


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

    def ind(x):
        # Return the index of the row in the data.
        return int(data[x][0])

    # Sanity check.
    assert ind(0) == r
    r = 0

    global bounds

    def label(left, right):
        if abs(real_query(ind(r), ind(left)) - real_query(ind(r), ind(right))) > ALPHA:
            mid = (left + right) // 2
            bounds[ind(mid)] = (
                real_query(ind(r), ind(left)),
                real_query(ind(r), ind(right)),
            )
            label(left, mid)
            label(mid, right)
        else:
            for x in range(left, right + 1):
                D[ind(r)][ind(x)] = real_query(ind(r), ind(left))

    label(0, len(data) - 1)

    for x in tqdm(range(len(data))):
        for y in range(len(data)):
            D[ind(x)][ind(y)] = abs(D[ind(r)][ind(x)] - D[ind(r)][ind(y)])

    np.save(f"{METRIC_DIR}/submetric_{ind(r)}.npy", D)
    return D


def max_merge(num_reps, data):
    """Max merge `num_reps` number of different alpha-submetrics."""

    reps = np.random.choice(len(data), num_reps, replace=False)
    D = np.zeros((len(data), len(data)))
    for r in reps:
        D = np.maximum(D, create_submetric(r, data))
    for submetric in submetrics.values():
        D = np.maximum(D, submetric)
    return D


def main():
    """Find an alpha-submetric."""

    with open(f"{DATA_DIR}/features.csv", "rb") as f:
        data = np.loadtxt(f, delimiter=",", skiprows=1, dtype=np.float32)

    # Open all submetrics.
    global submetrics
    for filename in os.listdir(METRIC_DIR):
        if filename.startswith("submetric_"):
            r = int(filename.split("_")[1].split(".")[0])
            submetrics[r] = np.load(f"{METRIC_DIR}/{filename}")

    global real_dists
    real_dists = np.full((len(data), len(data)), -1.0, dtype=np.float32)

    data = normalize(data, axis=0)
    data = np.insert(data, 0, np.arange(len(data)), axis=1)  # Add index column.
    D = max_merge(NUM_REPS - len(submetrics), data)
    np.save(f"{METRIC_DIR}/metric.npy", D)


if __name__ == "__main__":
    main()
