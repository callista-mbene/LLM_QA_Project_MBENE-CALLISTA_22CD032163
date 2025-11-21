from flask import Flask, render_template, request, jsonify
import os
import re
import google.generativeai as genai

app = Flask(__name__)

def preprocess_question(question):
    processed = question.lower()
    processed = re.sub(r'\s+', ' ', processed).strip()
    tokens = processed.split()
    return processed, tokens

def query_llm(question, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Please enter a valid question'}), 400
    
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        return jsonify({'error': 'API key not configured'}), 500
    
    processed_question, tokens = preprocess_question(question)
    answer = query_llm(question, api_key)
    
    return jsonify({
        'original_question': question,
        'processed_question': processed_question,
        'tokens': tokens,
        'answer': answer
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)