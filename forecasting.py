"""
forecasting.py
Expense Tracker — ML Forecasting Pipeline

Handles: data loading, preprocessing, weekly aggregation,
         outlier capping, model selection, and predictions.

Called by: app.py via run_forecast()
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine


# =============================================================
# DATABASE CONNECTION
# =============================================================

def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    return create_engine(db_url)


# =============================================================
# DATA LOADING
# =============================================================

def load_data(user_id):
    """Load expense transactions from Railway MySQL for a specific user."""
    engine = get_engine()
    query = "SELECT date, category, amount FROM expenses WHERE user_id = %(user_id)s"
    df = pd.read_sql(query, engine, params={"user_id": user_id})
    return df


# =============================================================
# PREPROCESSING
# =============================================================

def preprocess(df):
    """Parse dates, drop nulls, extract week index."""
    df['date'] = pd.to_datetime(df['date'])
    df.dropna(inplace=True)
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['year'] = df['date'].dt.year
    df['week_index'] = (df['year'] - df['year'].min()) * 52 + df['week']
    return df


# =============================================================
# WEEKLY AGGREGATION
# =============================================================

def aggregate_weekly(df):
    """Group daily transactions into weekly totals per category."""
    weekly = (
        df.groupby(['week_index', 'category'])['amount']
        .sum()
        .reset_index()
    )
    weekly.rename(columns={'amount': 'weekly_total'}, inplace=True)
    return weekly


# =============================================================
# OUTLIER CAPPING
# =============================================================

def cap_outliers(df, group_col, value_col):
    """
    Cap weekly totals exceeding 3x the category average.
    Prevents festivals or one-off spikes from skewing the model.
    """
    result = df.copy()
    for cat in result[group_col].unique():
        mask = result[group_col] == cat
        avg = result.loc[mask, value_col].mean()
        cap = avg * 3
        result.loc[mask, value_col] = result.loc[mask, value_col].clip(upper=cap)
    return result


# =============================================================
# DATA THRESHOLD CHECK
# =============================================================

def check_threshold(category_data):
    """
    Determine which model to use based on weeks of data available.
    - < 2 weeks  : insufficient
    - 2-3 weeks  : wma (Weighted Moving Average fallback)
    - 4+ weeks   : regression (Linear Regression)
    """
    weeks = len(category_data)
    if weeks < 2:
        return 'insufficient'
    elif weeks < 4:
        return 'wma'
    else:
        return 'regression'


# =============================================================
# MODELS
# =============================================================

def predict_linear_regression(category_data):
    """Train Linear Regression on weekly totals, predict next 4 weeks."""
    X = category_data['week_index'].values.reshape(-1, 1)
    y = category_data['weekly_total'].values

    model = LinearRegression()
    model.fit(X, y)

    last_week = category_data['week_index'].max()
    next_weeks = np.array([
        last_week + 1, last_week + 2,
        last_week + 3, last_week + 4
    ]).reshape(-1, 1)

    predictions = model.predict(next_weeks)
    predictions = np.clip(predictions, 0, None)
    return round(float(predictions.sum()), 2)


def predict_wma(category_data):
    """Weighted Moving Average fallback for sparse data (2-3 weeks)."""
    values = category_data['weekly_total'].values
    n = len(values)
    weights = np.arange(1, n + 1)
    wma = np.dot(weights, values) / weights.sum()
    return round(float(wma * 4), 2)


# =============================================================
# MAIN FORECAST RUNNER
# =============================================================

def run_forecast(user_id):
    """Full pipeline: load → preprocess → aggregate → cap → predict."""
    df = load_data(user_id)

    if df.empty:
        return {}

    df = preprocess(df)
    weekly = aggregate_weekly(df)
    weekly = cap_outliers(weekly, 'category', 'weekly_total')

    results = {}

    for cat in weekly['category'].unique():
        cat_data = weekly[weekly['category'] == cat].copy()
        tier = check_threshold(cat_data)

        if tier == 'insufficient':
            results[cat] = {
                'predicted_amount': None,
                'model_used': 'insufficient_data',
                'message': 'Keep logging expenses — not enough data yet.'
            }
        elif tier == 'wma':
            results[cat] = {
                'predicted_amount': predict_wma(cat_data),
                'model_used': 'weighted_moving_average',
                'message': 'Prediction based on recent spending (limited data).'
            }
        elif tier == 'regression':
            results[cat] = {
                'predicted_amount': predict_linear_regression(cat_data),
                'model_used': 'linear_regression',
                'message': 'Prediction based on your spending trend.'
            }

    total = sum(
        v['predicted_amount'] for v in results.values()
        if v['predicted_amount'] is not None
    )

    return {
        'total_predicted': round(total, 2),
        'predictions': results
    }