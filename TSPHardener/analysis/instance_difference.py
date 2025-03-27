import numpy as np
import pandas as pd
from analysis_util.load_json import load_full

def compute_sad_and_frobenius(initial, evolved):
    initial = np.array(initial)
    evolved = np.array(evolved)

    # Create a boolean mask to ignore diagonal (i.e., where i == j)
    n = initial.shape[0]
    mask = ~np.eye(n, dtype=bool)  # True for non-diagonal, False for diagonal

    # Apply the mask
    diff = evolved - initial
    diff_masked = diff[mask]

    sad = np.sum(np.abs(diff_masked))
    frob = np.sqrt(np.sum(diff_masked ** 2))
    return sad, frob


def collect_matrix_differences():
    data = load_full()
    records = []

    for entry in data:
        config = entry.get("configuration", {})
        results = entry.get("results", {})
        init_matrix = results.get("initial_matrix")

        if not init_matrix:
            continue

        # Track hard_instances differences
        hard_instances = results.get("hard_instances", {})
        for key, instance in hard_instances.items():
            evolved_matrix = instance.get("matrix")
            iterations = instance.get("iterations", None)

            if evolved_matrix and iterations is not None:
                sad, frob = compute_sad_and_frobenius(init_matrix, evolved_matrix)

                record = {
                    "generation": key,
                    "iterations": iterations,
                    "sad": sad,
                    "frobenius": frob,
                    **config
                }
                records.append(record)

        # Track last_matrix differences
        last_matrix = results.get("last_matrix", None)
        if last_matrix:
            print("analysing configuration", config)
            sad, frob = compute_sad_and_frobenius(init_matrix, last_matrix)
            record = {
                "generation": "last",
                "iterations": None,
                "sad": sad,
                "frobenius": frob,
                **config
            }
            records.append(record)

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    df = collect_matrix_differences()
    print(df.head())
    df.to_csv("matrix_differences.csv", index=False)
