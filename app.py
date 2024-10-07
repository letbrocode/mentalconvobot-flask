from flask import Flask, request, jsonify
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the fine-tuned Blenderbot model and tokenizer
model = BlenderbotForConditionalGeneration.from_pretrained('SomeUser675/mentalconvobot')
tokenizer = BlenderbotTokenizer.from_pretrained('SomeUser675/mentalconvobot')

# Function to generate a response
def generate_response(input_text, max_length=100, min_length=30, temperature=0.7, top_k=50, top_p=0.95):
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        response_ids = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=3,
            do_sample=True,
            length_penalty=1.0,
            early_stopping=True
        )
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    return response

# Define route for chatbot
@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.json.get('message')
    if not input_text:
        return jsonify({'error': 'No input text provided'}), 400

    response = generate_response(input_text)
    return jsonify({'response': response})

# Health check route
@app.route('/')
def health_check():
    return "Chatbot is running!"

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
