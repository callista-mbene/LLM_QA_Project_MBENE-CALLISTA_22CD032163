import os
import re
import string
import google.generativeai as genai

def preprocess_question(question):
    # lowercase
    processed = question.lower()
    # remove punctuation
    processed = processed.translate(str.maketrans('', '', string.punctuation))
    # remove extra spaces
    processed = re.sub(r'\s+', ' ', processed).strip()
    # tokenize
    tokens = processed.split()
    return ' '.join(tokens), tokens

def query_llm(question, api_key):
    try:
        genai.configure(api_key=api_key)

        # FIXED: updated model name
        model = genai.GenerativeModel("gemini-1.5-pro-latest")

        response = model.generate_content(question)
        
        # sometimes the response may not contain .text
        if hasattr(response, "text"):
            return response.text
        else:
            return str(response)

    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("=" * 60)
    print("NLP Question-and-Answering System Using LLM API")
    print("=" * 60)
    print()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set it using: export GEMINI_API_KEY='your-api-key'")
        return
    
    while True:
        print("\n" + "-" * 60)
        question = input("Enter your question (or 'quit' to exit): ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the Q&A system. Goodbye!")
            break
        
        if not question:
            print("Please enter a valid question.")
            continue
        
        print("\n[Processing...]")
        processed_question, tokens = preprocess_question(question)
        
        print(f"\nOriginal Question: {question}")
        print(f"Processed Question: {processed_question}")
        print(f"Tokens: {tokens}")
        
        print("\n[Querying LLM API...]")
        answer = query_llm(processed_question, api_key)
        
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print("=" * 60)

if __name__ == "__main__":
    main()
