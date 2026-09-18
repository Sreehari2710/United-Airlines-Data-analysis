"""
chat_app.py

A simple local, browser-based chat interface for the natural-language
data assistant (query_assistant.py). Instead of typing questions into the
terminal, this starts a small local web server so you can ask questions
through a normal-looking chat webpage in your browser.

This does NOT publish anything online -- it only runs on your own
computer, reachable only from this machine, at http://127.0.0.1:5000

Run from the project root:
    python ai_assistant/chat_app.py

Then open the printed URL (http://127.0.0.1:5000) in your browser.
Press Ctrl+C in the terminal to stop the server when you're done.
"""

from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

from query_assistant import answer_question

app = Flask(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.route("/")
def home():
    return send_from_directory(STATIC_DIR, "chat.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data or {}).get("question", "")
    if not question.strip():
        return jsonify({"answer": "Please type a question."})
    try:
        answer = answer_question(question)
    except Exception as e:
        answer = f"Something went wrong answering that: {e}"
    return jsonify({"answer": answer})


if __name__ == "__main__":
    print("Starting local chat UI for the United Airlines data assistant...")
    print("Open this in your browser:  http://127.0.0.1:5000")
    print("Press Ctrl+C to stop.\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
