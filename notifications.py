"""
notifications.py
Expense Tracker — Rule-Based Notification Engine

Handles 3 notification rules:
  Rule 1 — Budget Threshold Alert   (rule-based: 80% balance used)
  Rule 2 — Top Spending Category    (rule-based: highest category this week)
  Rule 3 — Anomaly Detection        (AI/stats: std deviation spike detection)

Called by: app.py via run_notifications()
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from sqlalchemy import create_engine, text

# =============================================================
# DATABASE CONNECTION
# =============================================================

DB_URL = "mysql+pymysql://root:lsNfxbtDHAVsZaAVGLPrMfiAfWEPpUYk@maglev.proxy.rlwy.net:28723/railway"

def get_engine():
    return create_engine(DB_URL)


# =============================================================
# DATA LOADING
# =============================================================

def load_data(user_id):
    """Load expense transactions from Railway for a specific user."""
    engine = get_engine()
    query = "SELECT date, category, amount FROM expenses WHERE user_id = %(user_id)s"
    df = pd.read_sql(query, engine, params={"user_id": user_id})
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_balance(user_id):
    """
    Load user's wallet balance from Railway.

    --- UPDATE WHEN deposits/wallet table is confirmed ---
    Currently returns hardcoded value.
    Replace with actual table/column name when ready.
    """
    # TODO: Update query when backend confirms table name
    # engine = get_engine()
    # with engine.connect() as conn:
    #     result = conn.execute(text("SELECT balance FROM wallet WHERE user_id = :uid"), {"uid": user_id})
    #     return float(result.fetchone()[0])
    return 1000.00


# =============================================================
# HELPER
# =============================================================

def get_this_week(df):
    """Returns only this week's expense records."""
    today = df['date'].max()
    week_start = today - timedelta(days=today.weekday())
    return df[df['date'] >= week_start]


# =============================================================
# RULE 1 — BUDGET THRESHOLD ALERT
# =============================================================

def check_budget_threshold(df, balance):
    """Triggers when total spending this month >= 80% of balance."""
    today = df['date'].max()
    this_month = df[
        (df['date'].dt.month == today.month) &
        (df['date'].dt.year == today.year)
    ]
    total_spent = this_month['amount'].sum()
    usage_pct = (total_spent / balance) * 100

    if usage_pct >= 80:
        return {
            'type': 'budget_alert',
            'severity': 'warning',
            'message': f"You have used {usage_pct:.0f}% of your balance this month — consider reducing spending."
        }
    return None


# =============================================================
# RULE 2 — TOP SPENDING CATEGORY THIS WEEK
# =============================================================

def check_top_category(df):
    """Finds the category with the highest total spending this week."""
    this_week = get_this_week(df)

    if this_week.empty:
        return None

    category_totals = this_week.groupby('category')['amount'].sum()
    top_category = category_totals.idxmax()
    top_amount = category_totals.max()

    return {
        'type': 'top_category',
        'severity': 'info',
        'message': f"{top_category} is your biggest expense this week at ${top_amount:.2f}."
    }


# =============================================================
# RULE 3 — ANOMALY DETECTION
# =============================================================

def check_spending_anomaly(df):
    """
    Std deviation based anomaly detection.
    Triggers when this week > mean + (2 x std) for any category.
    """
    alerts = []
    this_week = get_this_week(df)

    if this_week.empty:
        return alerts

    df = df.copy()
    df['week_index'] = (
        (df['date'].dt.year - df['date'].dt.year.min()) * 52 +
        df['date'].dt.isocalendar().week.astype(int)
    )

    weekly_totals = (
        df.groupby(['week_index', 'category'])['amount']
        .sum()
        .reset_index()
    )

    historical_stats = weekly_totals.groupby('category')['amount'].agg(['mean', 'std'])
    this_week_totals = this_week.groupby('category')['amount'].sum()

    for category, this_week_amount in this_week_totals.items():
        if category not in historical_stats.index:
            continue

        mean = historical_stats.loc[category, 'mean']
        std = historical_stats.loc[category, 'std']

        if pd.isna(std) or std == 0:
            continue

        threshold = mean + (2 * std)

        if this_week_amount > threshold:
            alerts.append({
                'type': 'anomaly_detected',
                'severity': 'warning',
                'message': f"Unusual spike detected in {category} spending this week — significantly higher than your historical average."
            })

    return alerts


# =============================================================
# MAIN NOTIFICATION RUNNER
# =============================================================

def run_notifications(user_id):
    """Runs all 3 rules and returns list of triggered alerts."""
    df = load_data(user_id)

    if df.empty:
        return []

    balance = load_balance(user_id)
    notifications = []

    budget_alert = check_budget_threshold(df, balance)
    if budget_alert:
        notifications.append(budget_alert)

    top_category = check_top_category(df)
    if top_category:
        notifications.append(top_category)

    anomaly_alerts = check_spending_anomaly(df)
    notifications.extend(anomaly_alerts)

    return notifications