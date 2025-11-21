import google.generativeai as genai

genai.configure(api_key="YOUR_KEY")

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("hello")

print(response.text)
