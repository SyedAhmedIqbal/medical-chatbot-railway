from flask import Flask, render_template, request, session, redirect, url_for
from chatbot import handle_chat  # Your chatbot logic

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session storage

# Store chat history in session
@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get("message")
    bot_response = handle_chat(user_message)

    # Update chat history
    session['chat_history'].append({"user": True, "text": user_message})
    session['chat_history'].append({"user": False, "text": bot_response})
    session.modified = True  # Save session changes

    return redirect(url_for('index'))

# Toggle dark mode using session storage
@app.route('/toggle_dark_mode', methods=['POST'])
def toggle_dark_mode():
    session['dark_mode'] = not session.get('dark_mode', False)
    return redirect(url_for('index'))

@app.route("/clear", methods=["POST"])
def clear():
    global context
    context = ""
    return redirect("/")  # redirects to fresh page
    
@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    session['chat_history'] = []
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
