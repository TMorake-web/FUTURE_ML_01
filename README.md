# Sales and Demand Forecasting

## Project Overview

This project uses machine learning to forecast daily retail sales based on historical sales data. A Random Forest regression model is developed and evaluated against a seasonal-naive baseline to assess its forecasting performance.

## Technologies

- Python
- Pandas
- Scikit-learn
- Matplotlib

## Dataset

The project uses the Store Sales — Time Series Forecasting dataset from Kaggle.

**Dataset:** https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

The training data contains 3,000,888 records covering the period from January 2013 to August 2017. Sales from all stores and product families are aggregated into a single daily time series for this implementation.

## Methodology

1. **Data preparation:** Converted data types, handled invalid records, and aggregated sales by date.
2. **Feature engineering:** Created calendar features, lag features, and rolling averages from historical sales.
3. **Model training:** Trained a Random Forest Regressor using historical observations.
4. **Model evaluation:** Compared predictions against actual sales over a 30-day validation period using MAE, RMSE, and WAPE.
5. **Forecasting:** Generated a 30-day forecast beyond the final date in the dataset.

## Model Evaluation

The models were evaluated on the same 30-day validation period.

| Metric | Random Forest | Seasonal-Naive Baseline |
|--------|--------------:|------------------------:|
| MAE    | 56,821.24     | 60,671.64               |
| RMSE   | 78,797.51     | 91,695.52               |
| WAPE   | 6.82%         | 7.28%                   |

The Random Forest model achieved lower errors across all three metrics. Its MAE was approximately 6.35% lower than that of the baseline on the selected validation period.

## Forecast

The model forecasts daily sales for the 30 days following the dataset's final date.

- **Forecast period:** 16 August – 14 September 2017
- **Total predicted sales:** Approximately 25.15 million
- **Average daily predicted sales:** Approximately 838,275

These predictions represent the period immediately following the historical dataset and are not forecasts for the current date.

## Visualisations and Outputs

The program saves the following files in the `outputs/` directory:

- `historical_sales.png` — historical daily sales trends.
- `validation_comparison.png` — actual sales compared with model predictions.
- `future_forecast.png` — historical sales and the forecast period.
- `validation_results.csv` — validation predictions and actual values.
- `model_comparison.csv` — evaluation metrics for both models.
- `sales_forecast.csv` — predicted sales for the forecast period.

## Installation and Usage

**Requirements:** Python and pip.

1. Clone the repository:

   
   git clone https://github.com/TMorake-web/FUTURE_ML_01.git
   cd FUTURE_ML_01
   

2. Install the dependencies:

   `
   pip install -r requirements.txt


3. Download `train.csv` from the Kaggle dataset source and place it in the project root.

4. Run the program:

   
   python main.py


The evaluation results, forecast, and visualisations will be generated in the `outputs/` directory.

## Limitations and Future Work

The current model forecasts aggregate sales and does not use store-level, product-family, promotion, or holiday information as predictors. Missing Christmas dates are filled with zero, an assumption that requires further verification. Evaluation on additional time periods and further feature engineering could improve the reliability of the forecasts.

## Project Information

**Program:** Future Interns Machine Learning Internship  
**Task:** Task 1 — Sales and Demand Forecasting  
**Model:** Random Forest Regressor  
**Status:** Initial implementation and evaluation completed
