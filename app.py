"""
app.py
Expense Tracker — Flask API

Routes only. No business logic here.
All logic lives in forecasting.py, notifications.py and chatbot.py

Endpoints:
  GET  /predict-expenses        → category spending forecasts
  GET  /check-notifications     → triggered alert notifications
  POST /chat                    → AI financial chatbot
  GET  /health                  → confirms server is running
"""

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request
from flask_cors import CORS
from forecasting import run_forecast
from notifications import run_notifications
from chatbot import get_chat_response

app = Flask(__name__)
CORS(app)


# =============================================================
# ROUTES
# =============================================================

@app.route('/predict-expenses', methods=['GET'])
def predict_expenses():
    """
    Returns next-month spending predictions per category.
    Requires: ?user_id=1
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
            'total_predicted': predictions['total_predicted'],
            'highest_category': predictions['highest_category'],
            'lowest_category': predictions['lowest_category'],
            'average_per_category': predictions['average_per_category'],
            'categories_tracked': predictions['categories_tracked'],
            'model_summary': predictions['model_summary'],
            'predictions': predictions['predictions']
        }), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/check-notifications', methods=['GET'])
def check_notifications():
    """
    Returns list of triggered notification alerts.
    Requires: ?user_id=1
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


@app.route('/chat', methods=['POST'])
def chat():
    """
    AI financial chatbot powered by Google Gemini.
    Answers questions about the user's spending data and general financial advice.

    Request body (JSON):
    {
        "user_id": 1,
        "message": "Am I spending too much?",
        "chat_history": []   (optional - for multi-turn conversation)
    }

    Response:
    {
        "status": "success",
        "reply": "Based on your predictions..."
    }
    """
    try:
        body = request.get_json()

        if not body:
            return jsonify({
                'status': 'error',
                'message': 'Request body must be JSON.'
            }), 400

        user_id = body.get('user_id')
        message = body.get('message')
        chat_history = body.get('chat_history', [])

        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'user_id is required.'
            }), 400

        if not message:
            return jsonify({
                'status': 'error',
                'message': 'message is required.'
            }), 400

        reply = get_chat_response(user_id, message, chat_history)

        return jsonify({
            'status': 'success',
            'reply': reply
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