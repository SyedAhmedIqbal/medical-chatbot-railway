import os
from dotenv import load_dotenv
import requests
import spacy
import logging
import subprocess

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)


try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
    nlp = spacy.load("en_core_web_md")
    
# Load medical terms for similarity checking
def load_medical_terms(file_path):
    try:
        with open(file_path, 'r') as file:
            return [nlp(line.strip()) for line in file.readlines() if line.strip()]
    except Exception as e:
        logging.error(f"Error loading medical terms: {e}")
        return []

core_terms = load_medical_terms('medical_term.txt')

# Groq API Configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_TOKEN = os.getenv("API_TOKEN")


GROQ_HEADERS = {
    "Authorization": f"Bearer {GROQ_API_TOKEN}",
    "Content-Type": "application/json"
}

#prompt for Groq
SYSTEM_PROMPT = (
"You are a knowledgeable and empathetic medical assistant AI. "
"Always tailor your response specifically to the user's described condition. "
"Do NOT repeat generic content for different symptoms — instead, diagnose the user's concern with precision.\n\n"
"Your output MUST:\n"
"- Start with a brief empathetic acknowledgement.\n"
"- Clearly list potential causes related only to the user input.\n"
"- Use <ul> and <li> tags for bullet lists.\n"
"- Use <strong> to highlight symptom names, diagnosis, or conditions.\n"
"- Avoid repeating the same recommendations unless truly applicable.\n"
"- Give actionable and relevant suggestions.\n"
"- Include a disclaimer reminding users to consult professionals.\n"
"- Do NOT include markdown or code formatting.\n"
"- Do NOT answer with vague or identical responses for different symptoms.\n"
)

#formatting for medical queries
def format_medical_response(query):
    return f"""
    <p><strong>I'm here to assist you with your health concern:</strong> {query}</p>
    
    <h3>Possible Causes:</h3>
    <ul>
        <li><strong>Tension or stress</strong></li>
        <li><strong>Dehydration</strong></li>
        <li><strong>Fatigue or lack of sleep</strong></li>
        <li><strong>Infections</strong> (e.g., cold, flu)</li>
        <li><strong>Environmental factors</strong> (e.g., allergens, weather changes)</li>
        <li><strong>Underlying medical conditions</strong> (e.g., chronic conditions)</li>
    </ul>
    
    <h3>Common Symptoms to Look Out For:</h3>
    <ul>
        <li>How long have you been experiencing these symptoms?</li>
        <li>Do you feel any other related symptoms (e.g., fever, nausea, fatigue)?</li>
    </ul>
    
    <h3>General Recommendations:</h3>
    <ul>
        <li><strong>Rest:</strong> Take a break and relax in a quiet, dark space.</li>
        <li><strong>Hydrate:</strong> Drink plenty of water or fluids to stay hydrated.</li>
        <li><strong>Healthy Eating:</strong> Try to eat balanced meals with sufficient nutrients.</li>
        <li><strong>Relaxation:</strong> Use relaxation techniques like deep breathing or meditation.</li>
    </ul>
    
    <p><strong>Disclaimer:</strong> These are general suggestions. Always consult with a healthcare professional for personalized medical advice, diagnosis, or treatment.</p>
    
    <p>Let me know if you need further assistance or have more specific questions!</p>
    """
# Template to format conversation history
TEMPLATE = '''
{context}
User: {question}
Note: The user is seeking help for a medical concern. Please analyze the unique condition mentioned and tailor your advice accordingly.
'''


# Call Groq API with prompt
def call_groq_model(prompt):
    try:
        payload = {
            "model": "llama3-70b-8192",  # or "mixtral-8x7b-32768"
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        response = requests.post(
            GROQ_API_URL,
            headers=GROQ_HEADERS,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        logging.error(f"Error calling Groq API: {e}")
        return "Sorry, I encountered an issue reaching the medical model."

# Determine if input is medical-related
def is_medical_question(user_input, threshold=0.5):
    if not core_terms:
        logging.warning("Medical terms list is empty.")
        return False
    user_doc = nlp(user_input.lower())
    return any(user_doc.similarity(term_doc) > threshold for term_doc in core_terms)

# Maintain conversation context
def update_context(context, user_input, response, max_turns=5):
    lines = context.strip().split('\n') if context else []
    lines += [f"User: {user_input}", f"AI: {response}"]
    return '\n'.join(lines[-2 * max_turns:])  # Last n turns

# Initialize conversation context
context = ""

def clear_chat():
    global context
    context = ""
    return "Chat history has been cleared. How can I assist you now?"


# Main chat handler
def handle_chat(user_input):
    global context
    if not user_input.strip():
        return "Please enter a valid message."
    
    if is_medical_question(user_input):
        prompt_input = TEMPLATE.format(context=context, question=user_input)
        response = call_groq_model(prompt_input)

        # Use fallback template only if model response is empty or invalid
        if not response or "Sorry" in response:
            response = format_medical_response(user_input)


    context = update_context(context, user_input, response)
    return response
