from pathlib import Path

import pandas as pd


# Find the project folder and its data file beside this script.
PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FILE = PROJECT_FOLDER / "train.csv"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs_v2"


def main():
    """Check the sales data and create weekly totals per store and family."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. Keep train.csv in the project folder."
        )

    print("Loading the sales data. This may take a little while...")

    # Read only the columns needed for weekly store and product-family sales.
    data = pd.read_csv(
        DATA_FILE,
        usecols=["date", "store_nbr", "family", "sales"],
    )

    rows_loaded = len(data)

    # Convert the date and sales columns into usable types.
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["sales"] = pd.to_numeric(data["sales"], errors="coerce")

    # Count rows with missing or invalid information before removing them.
    valid_rows = data.dropna(
        subset=["date", "store_nbr", "family", "sales"]
    ).copy()
    invalid_rows = rows_loaded - len(valid_rows)

    if valid_rows.empty:
        raise ValueError("No usable rows were found in train.csv.")

    first_date = valid_rows["date"].min()
    last_date = valid_rows["date"].max()

    # Put each date into a Monday-to-Sunday week.
    valid_rows["week_ending"] = (
        valid_rows["date"]
        + pd.to_timedelta(6 - valid_rows["date"].dt.dayofweek, unit="D")
    )

    # Sum the sales separately for each store, product family, and week.
    weekly_sales = (
        valid_rows
        .groupby(
            ["store_nbr", "family", "week_ending"],
            as_index=False,
            observed=True,
        )["sales"]
        .sum()
        .rename(columns={"sales": "weekly_sales"})
    )

    # Mark partial weeks at the very beginning or end of the data.
    weekly_sales["week_starting"] = (
        weekly_sales["week_ending"] - pd.Timedelta(days=6)
    )
    weekly_sales["full_calendar_week"] = (
        (weekly_sales["week_starting"] >= first_date)
        & (weekly_sales["week_ending"] <= last_date)
    )

    # Summarize how many weekly records each store-family pair has.
    weeks_per_series = weekly_sales.groupby(
        ["store_nbr", "family"],
        observed=True,
    ).size()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    weekly_file = OUTPUT_FOLDER / "weekly_sales.csv"
    weekly_sales.to_csv(weekly_file, index=False)

    # Save a short plain-language data summary.
    summary_lines = [
        "TASK 1 REBUILD: DATA SUMMARY",
        "",
        f"Rows read from train.csv: {rows_loaded:,}",
        f"Rows with missing or invalid required values: {invalid_rows:,}",
        f"Usable rows: {len(valid_rows):,}",
        f"First data date: {first_date.date()}",
        f"Last data date: {last_date.date()}",
        f"Number of stores: {valid_rows['store_nbr'].nunique():,}",
        f"Number of product families: {valid_rows['family'].nunique():,}",
        f"Store and product-family combinations: {len(weeks_per_series):,}",
        f"Weekly store-family records created: {len(weekly_sales):,}",
        f"Weekly sales file: {weekly_file}",
        "",
        "Sales are totaled separately for each store and product family.",
        "Partial calendar weeks at the dataset boundaries are marked in the CSV.",
    ]

    summary_text = "\n".join(summary_lines)
    summary_file = OUTPUT_FOLDER / "data_summary.txt"
    summary_file.write_text(summary_text, encoding="utf-8")

    print()
    print(summary_text)
    print()
    print(f"Saved the weekly data to: {weekly_file}")
    print(f"Saved the summary to: {summary_file}")


if __name__ == "__main__":
    main()