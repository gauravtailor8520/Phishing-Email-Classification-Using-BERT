from flask import Flask, request, render_template, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = Flask(__name__)

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('./phishing_model')
tokenizer = BertTokenizer.from_pretrained('./phishing_model')

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)
    return "Spam" if prediction.item() == 1 else "Ham"

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json
    text = data['text']
    prediction = predict(text)
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)

