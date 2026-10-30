from pathlib import Path

import pandas as pd

# Save charts without requiring a graphical window or Tcl/Tk.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================================
# FUTURE INTERNS - MACHINE LEARNING TASK 1
# SALES AND DEMAND FORECASTING
# ==========================================================

DATA_FILE = Path("train.csv")
OUTPUT_DIR = Path("outputs")

FORECAST_DAYS = 30
VALIDATION_DAYS = 30
RANDOM_STATE = 42


# ==========================================================
# 1. LOAD AND PREPARE THE DATA
# ==========================================================

def load_data(file_path):
    """Load the data and calculate total sales for each date."""

    print("\n" + "=" * 60)
    print("1. LOADING AND PREPARING DATA")
    print("=" * 60)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cannot find {file_path}. "
            "Place train.csv in your project folder."
        )

    df = pd.read_csv(
        file_path,
        usecols=["date", "sales"]
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["sales"] = pd.to_numeric(
        df["sales"],
        errors="coerce"
    )

    df = df.dropna(subset=["date", "sales"])

    daily_sales = (
        df.groupby("date")["sales"]
        .sum()
        .sort_index()
    )

    # Check for dates that have zero total sales BEFORE filling
    # any missing dates into the calendar.
    zero_sales_dates = daily_sales[daily_sales == 0]

    print("Usable original records:", len(df))
    print("Observed dates:", len(daily_sales))
    print("First observed date:", daily_sales.index.min().date())
    print("Last observed date:", daily_sales.index.max().date())
    print("Observed dates with zero total sales:",
          len(zero_sales_dates))

    if not zero_sales_dates.empty:
        print("\nFirst zero-sales dates to investigate:")
        print(zero_sales_dates.head(15).to_string())

    # Check for calendar dates missing from the dataset.
    full_dates = pd.date_range(
        start=daily_sales.index.min(),
        end=daily_sales.index.max(),
        freq="D"
    )

    missing_dates = full_dates.difference(daily_sales.index)

    print("\nCalendar dates missing from the dataset:",
          len(missing_dates))

    if len(missing_dates) > 0:
        print("First missing dates:")
        print(missing_dates[:15].tolist())

    # Fill missing calendar dates with zero for this initial model.
    # This assumption should be reviewed before real business use.
    daily_sales = daily_sales.reindex(
        full_dates,
        fill_value=0
    )

    daily_sales.index.name = "date"
    daily_sales.name = "sales"

    print("\nDaily observations after calendar preparation:",
          len(daily_sales))

    return daily_sales


# ==========================================================
# 2. CREATE FEATURES
# ==========================================================

def create_features(history, date):
    """Create predictors using only sales available before a date."""

    return {
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "day_of_week": date.day_of_week,
        "day_of_year": date.day_of_year,
        "week_of_year": int(date.isocalendar().week),

        "lag_1": history.iloc[-1],
        "lag_7": history.iloc[-7],
        "lag_14": history.iloc[-14],
        "lag_28": history.iloc[-28],

        "rolling_mean_7": history.iloc[-7:].mean(),
        "rolling_mean_28": history.iloc[-28:].mean(),
    }


def build_training_data(daily_sales):
    """Convert the time series into supervised ML examples."""

    records = []

    for position in range(28, len(daily_sales)):
        date = daily_sales.index[position]
        history = daily_sales.iloc[:position]

        features = create_features(history, date)

        features["target"] = daily_sales.iloc[position]
        features["date"] = date

        records.append(features)

    data = pd.DataFrame(records).set_index("date")

    feature_columns = [
        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
    ]

    X = data[feature_columns]
    y = data["target"]

    return X, y, feature_columns


# ==========================================================
# 3. RECURSIVE FORECASTING
# ==========================================================

def predict_forward(model, history, future_dates, feature_columns):
    """Predict multiple days without using their actual sales."""

    working_history = history.copy()
    predictions = []

    for date in future_dates:
        features = create_features(working_history, date)

        X_future = pd.DataFrame(
            [features],
            columns=feature_columns
        )

        prediction = float(model.predict(X_future)[0])
        prediction = max(0.0, prediction)

        predictions.append(prediction)

        # Use this prediction as history for the next forecast.
        working_history.loc[date] = prediction

    return pd.Series(
        predictions,
        index=future_dates,
        name="predicted_sales"
    )


def seasonal_naive_forecast(history, future_dates):
    """
    Baseline: repeat the most recent seven-day pattern.

    After each prediction, add it to history. This means the
    baseline does not use actual sales from the validation period.
    """

    working_history = history.copy()
    predictions = []

    for date in future_dates:
        prediction = float(working_history.iloc[-7])
        prediction = max(0.0, prediction)

        predictions.append(prediction)
        working_history.loc[date] = prediction

    return pd.Series(
        predictions,
        index=future_dates,
        name="baseline_predictions"
    )


# ==========================================================
# 4. EVALUATE AND COMPARE BOTH MODELS
# ==========================================================

def calculate_metrics(actual, predicted):
    """Calculate error metrics for a set of predictions."""

    mae = mean_absolute_error(actual, predicted)

    rmse = mean_squared_error(
        actual,
        predicted
    ) ** 0.5

    # WAPE avoids dividing each individual error by a possibly
    # zero actual value. It is still undefined if total actual
    # sales across the evaluation period are zero.
    total_actual = actual.abs().sum()

    if total_actual == 0:
        wape = float("nan")
    else:
        wape = (
            (actual - predicted).abs().sum()
            / total_actual
        ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
    }


def evaluate_model(daily_sales):
    """Compare Random Forest with a seasonal-naive baseline."""

    print("\n" + "=" * 60)
    print("2. MODEL TRAINING AND EVALUATION")
    print("=" * 60)

    if len(daily_sales) <= 28 + VALIDATION_DAYS:
        raise ValueError(
            "There are not enough daily observations "
            "for the selected validation period."
        )

    # Hold the final 30 days out for validation.
    split_date = daily_sales.index[-VALIDATION_DAYS]

    training_history = daily_sales.loc[
        daily_sales.index < split_date
    ]

    validation_actual = daily_sales.loc[
        daily_sales.index >= split_date
    ]

    X_train, y_train, feature_columns = build_training_data(
        training_history
    )

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    # Both approaches forecast the same validation dates.
    rf_predictions = predict_forward(
        model,
        training_history,
        validation_actual.index,
        feature_columns
    )

    baseline_predictions = seasonal_naive_forecast(
        training_history,
        validation_actual.index
    )

    rf_metrics = calculate_metrics(
        validation_actual,
        rf_predictions
    )

    baseline_metrics = calculate_metrics(
        validation_actual,
        baseline_predictions
    )

    metrics_table = pd.DataFrame(
        {
            "Random Forest": rf_metrics,
            "Seasonal Naive Baseline": baseline_metrics,
        }
    ).T

    print("\nVALIDATION RESULTS")
    print("-" * 60)
    print("Training observations:", len(X_train))
    print("Validation days:", len(validation_actual))
    print("\n", metrics_table.round(2).to_string())

    rf_mae = rf_metrics["MAE"]
    baseline_mae = baseline_metrics["MAE"]

    if baseline_mae > 0:
        improvement = (
            (baseline_mae - rf_mae) / baseline_mae
        ) * 100

        print(f"\nRandom Forest MAE improvement: {improvement:.2f}%")

        if improvement > 0:
            print("Interpretation: Random Forest has lower MAE.")
        elif improvement < 0:
            print("Interpretation: The baseline has lower MAE.")
        else:
            print("Interpretation: Both models have equal MAE.")
    else:
        print("\nBaseline MAE is zero; percentage improvement "
              "cannot be calculated.")

    validation_results = pd.DataFrame({
        "actual_sales": validation_actual,
        "random_forest_prediction": rf_predictions,
        "baseline_prediction": baseline_predictions,
    })

    return (
        validation_results,
        metrics_table,
        feature_columns,
        model
    )


# ==========================================================
# 5. TRAIN THE FINAL MODEL ON ALL AVAILABLE DATA
# ==========================================================

def train_final_model(daily_sales, feature_columns):
    """Train a final Random Forest using all historical examples."""

    print("\n" + "=" * 60)
    print("3. TRAINING FINAL MODEL")
    print("=" * 60)

    X, y, _ = build_training_data(daily_sales)

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("Training on all historical observations...")
    model.fit(X, y)

    print("Final model trained.")
    print("Training observations:", len(X))

    return model


# ==========================================================
# 6. GENERATE THE NEXT 30 DAYS AFTER THE DATASET ENDS
# ==========================================================

def create_forecast(model, daily_sales, feature_columns):
    """Generate a 30-day forecast beyond the final dataset date."""

    print("\n" + "=" * 60)
    print("4. GENERATING 30-DAY FORECAST")
    print("=" * 60)

    future_dates = pd.date_range(
        start=daily_sales.index.max() + pd.Timedelta(days=1),
        periods=FORECAST_DAYS,
        freq="D"
    )

    forecast = predict_forward(
        model,
        daily_sales,
        future_dates,
        feature_columns
    )

    print("\nFORECAST RESULTS")
    print(forecast.to_string())

    print(f"\nForecast total: {forecast.sum():,.2f}")
    print(f"Average daily forecast: {forecast.mean():,.2f}")

    return forecast


# ==========================================================
# 7. SAVE RESULTS AND CHARTS
# ==========================================================

def save_outputs(daily_sales, validation, metrics, forecast):
    """Save CSV results and visualisations to the outputs folder."""

    print("\n" + "=" * 60)
    print("5. SAVING RESULTS AND CHARTS")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    validation.to_csv(
        OUTPUT_DIR / "validation_results.csv",
        index_label="date"
    )

    metrics.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index_label="model"
    )

    forecast.to_frame().to_csv(
        OUTPUT_DIR / "sales_forecast.csv",
        index_label="date"
    )

    # Chart 1: Historical daily sales.
    plt.figure(figsize=(12, 5))
    plt.plot(daily_sales.index, daily_sales.values, linewidth=1)
    plt.title("Historical Daily Sales")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "historical_sales.png", dpi=150)
    plt.close()

    # Chart 2: Actual sales compared with both models.
    plt.figure(figsize=(12, 5))
    plt.plot(
        validation.index,
        validation["actual_sales"],
        label="Actual sales"
    )
    plt.plot(
        validation.index,
        validation["random_forest_prediction"],
        label="Random Forest"
    )
    plt.plot(
        validation.index,
        validation["baseline_prediction"],
        label="Seasonal naive baseline"
    )
    plt.title("Validation: Actual vs Model Predictions")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "validation_comparison.png",
        dpi=150
    )
    plt.close()

    # Chart 3: Recent history and future forecast.
    recent_history = daily_sales.tail(90)

    plt.figure(figsize=(12, 5))
    plt.plot(
        recent_history.index,
        recent_history.values,
        label="Historical sales"
    )
    plt.plot(
        forecast.index,
        forecast.values,
        label="30-day forecast"
    )
    plt.axvline(
        daily_sales.index.max(),
        linestyle="--",
        label="Forecast starts"
    )
    plt.title("Daily Sales Forecast")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "future_forecast.png",
        dpi=150
    )
    plt.close()

    print("Output folder:", OUTPUT_DIR.resolve())
    print("Saved: validation_results.csv")
    print("Saved: model_comparison.csv")
    print("Saved: sales_forecast.csv")
    print("Saved: historical_sales.png")
    print("Saved: validation_comparison.png")
    print("Saved: future_forecast.png")


# ==========================================================
# 8. RUN THE COMPLETE PROJECT
# ==========================================================

def main():
    """Run data preparation, training, evaluation and forecasting."""

    daily_sales = load_data(DATA_FILE)

    (
        validation_results,
        metrics_table,
        feature_columns,
        validation_model
    ) = evaluate_model(daily_sales)

    # Train a final model for the 30-day forecast.
    final_model = train_final_model(
        daily_sales,
        feature_columns
    )

    forecast = create_forecast(
        final_model,
        daily_sales,
        feature_columns
    )

    save_outputs(
        daily_sales,
        validation_results,
        metrics_table,
        forecast
    )

    print("\n" + "=" * 60)
    print("FORECASTING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("Compare the model metrics in outputs/model_comparison.csv")
    print("Review the charts and CSV files in the outputs folder.")


if __name__ == "__main__":
    main()