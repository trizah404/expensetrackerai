"""
chatbot.py
Expense Tracker — AI Financial Chatbot

Uses Google Gemini to answer user questions about their spending.
Context is built from the user's actual ML predictions so every
response is personalised rather than generic.

Called by: app.py via get_chat_response()
"""

import os
import time
from google import genai
from forecasting import run_forecast


# =============================================================
# GEMINI SETUP
# =============================================================

def get_gemini_client():
    """Configure and return the Gemini client."""
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


# =============================================================
# CONTEXT BUILDER
# =============================================================

def build_context(user_id):
    """
    Fetches the user's ML predictions and formats them
    into a context string for Gemini to use when answering.
    Returns None if no data is available.
    """
    try:
        data = run_forecast(user_id)

        if not data or not data.get('predictions'):
            return None

        predictions = data['predictions']
        total = data.get('total_predicted', 0)
        highest = data.get('highest_category', {})
        lowest = data.get('lowest_category', {})
        avg = data.get('average_per_category', 0)
        categories_tracked = data.get('categories_tracked', 0)

        category_lines = []
        for cat, v in predictions.items():
            amt = v.get('predicted_amount')
            pct = v.get('percentage_of_total', 0)
            if amt is not None:
                category_lines.append(
                    f"  - {cat}: Rs {amt:.2f} ({pct}% of total budget)"
                )

        context = f"""
The user's predicted spending for next month:
- Total predicted: Rs {total:.2f}
- Highest spending category: {highest.get('name', 'N/A')} at Rs {highest.get('amount', 0):.2f}
- Lowest spending category: {lowest.get('name', 'N/A')} at Rs {lowest.get('amount', 0):.2f}
- Average per category: Rs {avg:.2f}
- Number of categories tracked: {categories_tracked}

Breakdown by category:
{chr(10).join(category_lines)}
"""
        return context.strip()

    except Exception:
        return None


# =============================================================
# SYSTEM PROMPT
# =============================================================

SYSTEM_PROMPT = """
You are a helpful and friendly financial assistant built into an expense tracking application called Expense Tracker.

Your role is to:
1. Answer questions about the user's spending patterns and predictions using the data provided.
2. Give practical, personalised financial advice based on their actual spending data.
3. Help users understand how to use the Expense Tracker app (adding expenses, reading predictions, understanding notifications).
4. Answer general financial questions in a friendly, approachable way.

Guidelines:
- Always be encouraging and supportive, never judgmental about spending habits.
- Keep responses concise — 2 to 4 sentences unless the user asks for more detail.
- When spending data is available, always reference it specifically rather than giving generic advice.
- Use Rs as the currency symbol.
- Do not make up data — only use the figures provided to you.
- If no spending data is available, still answer helpfully using general financial advice.
"""


# =============================================================
# MAIN CHAT FUNCTION
# =============================================================

def get_chat_response(user_id, message, chat_history=None):
    """
    Generates a personalised response to the user's message.
    Retries up to 3 times if the model is temporarily unavailable.
    """
    context = build_context(user_id)

    if context:
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser spending data:\n{context}\n\nUser question: {message}"
    else:
        full_prompt = f"{SYSTEM_PROMPT}\n\nNote: No spending data available for this user yet.\n\nUser question: {message}"

    for attempt in range(3):
        try:
            client = get_gemini_client()
            response = client.models.generate_content(
                model="models/gemini-2.5-flash-lite",
                contents=full_prompt
            )
            return response.text.strip()

        except Exception as e:
            error_msg = str(e)
            if attempt < 2 and ('503' in error_msg or 'UNAVAILABLE' in error_msg or '429' in error_msg):
                time.sleep(3)
                continue
            return "I'm currently unavailable. Please try again in a moment."