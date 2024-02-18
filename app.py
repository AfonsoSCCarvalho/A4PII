from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model
tokenizer = AutoTokenizer.from_pretrained("Isotonic/distilbert_finetuned_ai4privacy_v2")
model = AutoModelForTokenClassification.from_pretrained("Isotonic/distilbert_finetuned_ai4privacy_v2")
pipe = pipeline("token-classification", model=model, tokenizer=tokenizer)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    text = data['text']
    results = pipe(text)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
