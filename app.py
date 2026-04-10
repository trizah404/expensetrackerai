"""
app.py
Expense Tracker — Flask API

Routes only. No business logic here.
All logic lives in forecasting.py and notifications.py.

Endpoints:
  GET /predict-expenses      → category spending forecasts
  GET /check-notifications   → triggered alert notifications
  GET /health                → confirms server is running
"""

from flask import Flask, jsonify, request
from forecasting import run_forecast
from notifications import run_notifications

app = Flask(__name__)


# =============================================================
# ROUTES
# =============================================================

@app.route('/predict-expenses', methods=['GET'])
def predict_expenses():
    """
    Returns next-month spending predictions per category.

    Requires: ?user_id=1
    Example:  /predict-expenses?user_id=1

    Response:
    {
        "status": "success",
        "predictions": {
            "Food":      { "predicted_amount": 131.60, "model_used": "linear_regression" },
            "Transport": { "predicted_amount": 21.64,  "model_used": "linear_regression" }
        }
    }
    """
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'user_id is required. Example: /predict-expenses?user_id=1'
            }), 400

        predictions = run_forecast(user_id)
        return jsonify({
            'status': 'success',
            'predictions': predictions
        }), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/check-notifications', methods=['GET'])
def check_notifications():
    """
    Returns list of triggered notification alerts.

    Requires: ?user_id=1
    Example:  /check-notifications?user_id=1

    Response:
    {
        "status": "success",
        "count": 2,
        "notifications": [
            { "type": "budget_alert", "severity": "warning", "message": "..." },
            { "type": "top_category", "severity": "info",    "message": "..." }
        ]
    }
    """
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'user_id is required. Example: /check-notifications?user_id=1'
            }), 400

        notifications = run_notifications(user_id)
        return jsonify({
            'status': 'success',
            'count': len(notifications),
            'notifications': notifications
        }), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Quick check to confirm the API is running."""
    return jsonify({'status': 'running'}), 200


# =============================================================
# RUN
# =============================================================

if __name__ == '__main__':
    app.run(debug=True, port=5000)