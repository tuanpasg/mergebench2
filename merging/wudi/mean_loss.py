#!/usr/bin/env python3
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean loss for each numeric loss column in a CSV file."
    )
    parser.add_argument("csv_path", help="Path to input CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    # Select only numeric columns, e.g. loss_0, loss_1, loss_2, ...
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        print("No numeric columns found.")
        return

    print("Mean loss per column:")
    print("-" * 40)

    for col, mean_value in numeric_df.mean().items():
        print(f"{col}: {mean_value:.10f}")


if __name__ == "__main__":
    main()