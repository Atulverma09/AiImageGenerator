# app.py
import os
import requests
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from io import BytesIO

load_dotenv() # .env file se environment variables load karega

app = Flask(__name__)
CORS(app) # CORS ko enable karega, taaki frontend se request aa sake

# Hugging Face API ka URL aur Token
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

@app.route("/")
def home():
    return "Backend is running!"

@app.route("/generate", methods=["POST"])
def generate_image():
    try:
        # Frontend se JSON data lo
        data = request.get_json()
        prompt = data.get("prompt")

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Hugging Face API ko call karo
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        
        # Agar API se error aaye
        if response.status_code != 200:
            return jsonify({"error": "Failed to generate image", "details": response.json()}), 500

        # API se image bytes milenge, usko file ki tarah bhejo
        image_bytes = response.content
        return send_file(BytesIO(image_bytes), mimetype='image/jpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)