"""Prepare a quick-validation parquet using trailing rows or random sampling.

Existing output files are read and reused without modification.
"""

import argparse
from pathlib import Path


def positive_int(value):
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("num-samples must be greater than zero")
    return count


def prepare_data(input_path, output_path, num_samples, sampling="tail"):
    import pandas as pd

    if num_samples <= 0:
        raise ValueError("num_samples must be greater than zero")
    source = Path(input_path).expanduser()
    output = Path(output_path).expanduser()

    if output.exists():
        selected = pd.read_parquet(output)
        print(f"Reusing existing validation set: {output}")
        print(f"Rows: {len(selected)} (requested count does not modify an existing file)")
        return selected

    if source.resolve() == output.resolve():
        raise ValueError("Input and output paths must be different")
    data = pd.read_parquet(source)
    if data.empty:
        raise ValueError(f"Source dataset is empty: {source}")
    if sampling == "random":
        selected = data.sample(n=min(num_samples, len(data)))
    elif sampling == "tail":
        selected = data.tail(num_samples)
    else:
        raise ValueError(f"Unknown sampling method: {sampling}")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output, index=False)
    print(f"Source rows: {len(data)}")
    print(f"Saved {len(selected)} rows ({sampling}) to: {output}")
    if len(selected) < num_samples:
        print(f"Source has fewer than {num_samples} rows; saved all available rows")
    print(f"Columns: {selected.columns.tolist()}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source parquet file")
    parser.add_argument("--output", required=True, help="Quick-validation parquet file; reused if it exists")
    parser.add_argument("--num-samples", required=True, type=positive_int, help="Number of rows to select")
    parser.add_argument("--sampling", choices=["tail", "random"], default="tail", help="Sampling method (default: tail)")
    args = parser.parse_args()
    prepare_data(args.input, args.output, args.num_samples, args.sampling)


if __name__ == "__main__":
    main()
