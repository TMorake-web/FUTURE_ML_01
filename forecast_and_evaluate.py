"""
Task 1: Weekly sales planning by store and product family.

This version trains one shared model across the store-family records.
It compares that model with a simple seasonal baseline, then forecasts
the next 12 weeks for each store and product family.

The dataset does not include stock levels or supplier lead times, so these
sales forecasts cannot calculate exact reorder amounts.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save the chart to a file without opening a window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# --------------------------- Project settings -----------------------------

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs_v2"
WEEKLY_FILE = OUTPUT_DIR / "weekly_sales.csv"

DATE_COLUMN = "week_starting"
SALES_COLUMN = "weekly_sales"
GROUP_COLUMNS = ["store_nbr", "family"]

FORECAST_WEEKS = 12
VALIDATION_WEEKS = 4
VALIDATION_CUTOFFS = 3
RANDOM_STATE = 42


# -------------------------- Feature creation ------------------------------

def create_features(data):
    """Create calendar and historical-sales features for each series."""
    result = data.sort_values(
        GROUP_COLUMNS + [DATE_COLUMN]
    ).copy()

    dates = result[DATE_COLUMN]

    # Calendar features describe the time of year.
    result["month"] = dates.dt.month
    result["quarter"] = dates.dt.quarter
    result["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    result["week_sin"] = np.sin(
        2 * np.pi * result["week_of_year"] / 52.18
    )
    result["week_cos"] = np.cos(
        2 * np.pi * result["week_of_year"] / 52.18
    )
    result["month_sin"] = np.sin(
        2 * np.pi * result["month"] / 12
    )
    result["month_cos"] = np.cos(
        2 * np.pi * result["month"] / 12
    )

    grouped_sales = result.groupby(
        GROUP_COLUMNS, sort=False
    )[SALES_COLUMN]

    # Earlier sales from this same store and product family.
    for lag in (1, 2, 4, 13, 26, 52):
        result[f"lag_{lag}"] = grouped_sales.shift(lag)

    # Recent average sales, using only weeks before the row being predicted.
    for window in (4, 13, 52):
        result[f"rolling_mean_{window}"] = (
            result.groupby(GROUP_COLUMNS, sort=False)[SALES_COLUMN]
            .transform(
                lambda values: values.shift(1)
                .rolling(window, min_periods=1)
                .mean()
            )
        )

    # Average sales for this series up to, but not including, this week.
    cumulative_sales = grouped_sales.cumsum()
    previous_row_count = result.groupby(
        GROUP_COLUMNS, sort=False
    ).cumcount()

    result["series_average"] = (
        (cumulative_sales - result[SALES_COLUMN])
        / previous_row_count.replace(0, np.nan)
    )

    return result


FEATURE_COLUMNS = [
    "month",
    "quarter",
    "week_of_year",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_13",
    "lag_26",
    "lag_52",
    "rolling_mean_4",
    "rolling_mean_13",
    "rolling_mean_52",
    "series_average",
]


def create_model():
    """Make the shared forecasting model."""
    return HistGradientBoostingRegressor(
        max_iter=80,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


def seasonal_baseline(row):
    """Use last year's matching week, or a recent average if unavailable."""
    if pd.notna(row["lag_52"]):
        return max(0.0, float(row["lag_52"]))

    if pd.notna(row["rolling_mean_4"]):
        return max(0.0, float(row["rolling_mean_4"]))

    return 0.0


def calculate_wape(actual, predicted):
    """Calculate weighted absolute percentage error."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    total_sales = np.abs(actual_values).sum()

    if total_sales == 0:
        return 0.0

    return float(
        np.abs(actual_values - predicted_values).sum() / total_sales
    )


# ---------------------------- Model fitting -------------------------------

def train_model(training_rows):
    """Train one model using all eligible store-family records."""
    usable = training_rows.dropna(
        subset=FEATURE_COLUMNS + [SALES_COLUMN]
    )

    if usable.empty:
        raise ValueError(
            "There are no training rows with enough sales history."
        )

    model = create_model()
    model.fit(
        usable[FEATURE_COLUMNS],
        usable[SALES_COLUMN],
    )
    return model


# ------------------------------ Backtesting -------------------------------

def run_backtests(feature_data, last_complete_week):
    """Evaluate the model on three historical four-week periods."""
    all_actual = []
    all_model_predictions = []
    all_baseline_predictions = []

    # Each cutoff is separated by the four-week evaluation window.
    cutoffs = [
        last_complete_week
        - pd.Timedelta(weeks=VALIDATION_WEEKS * offset)
        for offset in range(VALIDATION_CUTOFFS, 0, -1)
    ]

    print("Checking historical periods with one shared model...")

    for number, cutoff in enumerate(cutoffs, start=1):
        end_week = cutoff + pd.Timedelta(weeks=VALIDATION_WEEKS)

        training_rows = feature_data.loc[
            feature_data[DATE_COLUMN] < cutoff
        ].dropna(subset=FEATURE_COLUMNS + [SALES_COLUMN])

        validation_rows = feature_data.loc[
            (feature_data[DATE_COLUMN] >= cutoff)
            & (feature_data[DATE_COLUMN] < end_week)
        ].dropna(subset=FEATURE_COLUMNS + [SALES_COLUMN])

        print(
            f"Period {number} of {len(cutoffs)}: "
            f"{cutoff.date()} to before {end_week.date()}"
        )

        if training_rows.empty or validation_rows.empty:
            print("  Skipping this period because it has no usable rows.")
            continue

        model = train_model(training_rows)
        predictions = np.maximum(
            0,
            model.predict(validation_rows[FEATURE_COLUMNS]),
        )

        baselines = validation_rows.apply(
            seasonal_baseline,
            axis=1,
        ).to_numpy(dtype=float)

        all_actual.extend(validation_rows[SALES_COLUMN].to_numpy())
        all_model_predictions.extend(predictions)
        all_baseline_predictions.extend(baselines)

    if not all_actual:
        raise RuntimeError(
            "No historical predictions were evaluated. "
            "Please send the full terminal output."
        )

    return (
        np.asarray(all_actual, dtype=float),
        np.asarray(all_model_predictions, dtype=float),
        np.asarray(all_baseline_predictions, dtype=float),
        cutoffs,
    )


# -------------------------- Future forecasting ----------------------------

def make_future_row(history, group_values, target_week):
    """Build one forecast row using that series' available past sales."""
    row = {
        "store_nbr": group_values[0],
        "family": group_values[1],
        DATE_COLUMN: target_week,
    }

    month = target_week.month
    week_of_year = int(target_week.isocalendar().week)

    row["month"] = month
    row["quarter"] = target_week.quarter
    row["week_of_year"] = week_of_year
    row["week_sin"] = np.sin(2 * np.pi * week_of_year / 52.18)
    row["week_cos"] = np.cos(2 * np.pi * week_of_year / 52.18)
    row["month_sin"] = np.sin(2 * np.pi * month / 12)
    row["month_cos"] = np.cos(2 * np.pi * month / 12)

    sales_by_week = {
        pd.Timestamp(date): float(sales)
        for date, sales in zip(
            history[DATE_COLUMN],
            history[SALES_COLUMN],
        )
    }

    for lag in (1, 2, 4, 13, 26, 52):
        past_week = target_week - pd.Timedelta(weeks=lag)
        row[f"lag_{lag}"] = sales_by_week.get(past_week, np.nan)

    past_sales = history[SALES_COLUMN].to_numpy(dtype=float)

    for window in (4, 13, 52):
        recent = past_sales[-window:]
        row[f"rolling_mean_{window}"] = (
            float(np.mean(recent)) if len(recent) else np.nan
        )

    row["series_average"] = (
        float(np.mean(past_sales)) if len(past_sales) else np.nan
    )

    return row


def create_future_forecast(model, complete_data, last_complete_week):
    """Forecast each store-family series 12 weeks into the future."""
    histories = {}

    for group_values, series in complete_data.groupby(
        GROUP_COLUMNS,
        sort=False,
    ):
        histories[group_values] = (
            series[[DATE_COLUMN, SALES_COLUMN]]
            .sort_values(DATE_COLUMN)
            .copy()
        )

    forecast_rows = []

    print("Creating the 12-week forecasts...")

    for step in range(1, FORECAST_WEEKS + 1):
        target_week = last_complete_week + pd.Timedelta(weeks=step)
        future_rows = []

        for group_values, history in histories.items():
            future_rows.append(
                make_future_row(history, group_values, target_week)
            )

        future_frame = pd.DataFrame(future_rows)
        future_frame[FEATURE_COLUMNS] = future_frame[
            FEATURE_COLUMNS
        ].fillna(0)

        predictions = np.maximum(
            0,
            model.predict(future_frame[FEATURE_COLUMNS]),
        )

        for position, (group_values, history) in enumerate(
            histories.items()
        ):
            predicted_sales = float(predictions[position])

            forecast_rows.append(
                {
                    "store_nbr": group_values[0],
                    "family": group_values[1],
                    "week_starting": target_week.date().isoformat(),
                    "forecast_sales": predicted_sales,
                }
            )

            new_history_row = pd.DataFrame(
                {
                    DATE_COLUMN: [target_week],
                    SALES_COLUMN: [predicted_sales],
                }
            )
            histories[group_values] = pd.concat(
                [history, new_history_row],
                ignore_index=True,
            )

    return pd.DataFrame(forecast_rows)


# --------------------------------- Main -----------------------------------

def main():
    """Load data, evaluate the model, forecast, and save results."""
    if not WEEKLY_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {WEEKLY_FILE}. "
            "Run forecast_by_store_family.py first."
        )

    print("Loading the prepared weekly sales data...")

    data = pd.read_csv(
        WEEKLY_FILE,
        parse_dates=["week_starting", "week_ending"],
    )

    required_columns = {
        "store_nbr",
        "family",
        "week_starting",
        "week_ending",
        "weekly_sales",
        "full_calendar_week",
    }
    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"weekly_sales.csv is missing columns: {sorted(missing)}"
        )

    data["full_calendar_week"] = (
        data["full_calendar_week"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    data[SALES_COLUMN] = pd.to_numeric(
        data[SALES_COLUMN],
        errors="coerce",
    )
    data = data.dropna(
        subset=GROUP_COLUMNS + [DATE_COLUMN, SALES_COLUMN]
    )

    # Only complete weeks are used in the model and its evaluation.
    complete_data = data.loc[data["full_calendar_week"]].copy()
    complete_data = complete_data.sort_values(
        GROUP_COLUMNS + [DATE_COLUMN]
    )

    last_complete_week = complete_data[DATE_COLUMN].max()

    print(f"Complete weekly records: {len(complete_data):,}")
    print(f"Last complete week: {last_complete_week.date()}")
    print("Preparing the model features...")

    feature_data = create_features(complete_data)

    # Evaluate on earlier periods that the model did not train on.
    (
        actual,
        model_predictions,
        baseline_predictions,
        cutoffs,
    ) = run_backtests(feature_data, last_complete_week)

    model_mae = mean_absolute_error(actual, model_predictions)
    baseline_mae = mean_absolute_error(actual, baseline_predictions)
    model_wape = calculate_wape(actual, model_predictions)
    baseline_wape = calculate_wape(actual, baseline_predictions)

    if baseline_mae > 0:
        improvement = (
            100 * (baseline_mae - model_mae) / baseline_mae
        )
    else:
        improvement = 0.0

    comparison = pd.DataFrame(
        [
            {
                "approach": "Shared machine-learning model",
                "MAE": model_mae,
                "WAPE_percent": model_wape * 100,
                "evaluated_weekly_records": len(actual),
            },
            {
                "approach": "Seasonal baseline",
                "MAE": baseline_mae,
                "WAPE_percent": baseline_wape * 100,
                "evaluated_weekly_records": len(actual),
            },
        ]
    )
    comparison.to_csv(
        OUTPUT_DIR / "model_comparison_v2.csv",
        index=False,
    )

    # Train once more using all available history for the future forecast.
    print("Training the final shared model...")
    final_training_rows = feature_data.dropna(
        subset=FEATURE_COLUMNS + [SALES_COLUMN]
    )
    final_model = train_model(final_training_rows)

    forecast = create_future_forecast(
        final_model,
        complete_data,
        last_complete_week,
    )
    forecast.to_csv(
        OUTPUT_DIR / "sales_forecast_v2.csv",
        index=False,
    )

    # Create a simple chart where shorter error bars mean better forecasts.
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        ["Machine-learning model", "Seasonal baseline"],
        [model_mae, baseline_mae],
        color=["#2878B5", "#9AA6B2"],
    )
    axis.set_title("Forecast error on historical weeks")
    axis.set_ylabel("Average error in sales (lower is better)")
    axis.bar_label(bars, fmt="%.0f")
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "forecast_evaluation_v2.png",
        dpi=160,
    )
    plt.close(figure)

    report_lines = [
        "TASK 1 REBUILD — WEEKLY SALES PLANNING",
        "",
        f"Complete weekly records evaluated: {len(actual):,}",
        f"Historical evaluation periods: {len(cutoffs)}",
        f"Machine-learning model MAE: {model_mae:,.2f}",
        f"Seasonal baseline MAE: {baseline_mae:,.2f}",
        f"Machine-learning model WAPE: {model_wape * 100:.2f}%",
        f"Seasonal baseline WAPE: {baseline_wape * 100:.2f}%",
        f"MAE improvement compared with baseline: {improvement:+.2f}%",
        "",
        "How to read the results:",
        "- Lower error means predictions were closer to actual sales.",
        "- Positive improvement means the model had lower error.",
        "- Negative improvement means the baseline performed better.",
        "",
        "Limits:",
        "- Forecasts estimate weekly sales by store and product family.",
        "- The dataset does not include stock levels or supplier lead times.",
        "- Sales forecasts cannot tell us exact amounts to order.",
        "- Use the results to support planning and human decisions.",
    ]

    report_path = OUTPUT_DIR / "forecast_summary_v2.txt"
    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nHISTORICAL CHECK RESULTS")
    print(f"Records evaluated: {len(actual):,}")
    print(f"Model MAE: {model_mae:,.2f}")
    print(f"Baseline MAE: {baseline_mae:,.2f}")
    print(f"Model WAPE: {model_wape * 100:.2f}%")
    print(f"Baseline WAPE: {baseline_wape * 100:.2f}%")
    print(f"Model MAE improvement: {improvement:+.2f}%")
    print("\nSaved files:")
    print(f"Forecast: {OUTPUT_DIR / 'sales_forecast_v2.csv'}")
    print(f"Comparison: {OUTPUT_DIR / 'model_comparison_v2.csv'}")
    print(f"Chart: {OUTPUT_DIR / 'forecast_evaluation_v2.png'}")
    print(f"Plain-English report: {report_path}")


if __name__ == "__main__":
    main()