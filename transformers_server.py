#!/usr/bin/env python
"""
transformers_server.py

A simple Flask API server for language model inference using the transformers library.
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Global variables to store model and tokenizer
model = None
tokenizer = None
device = None

@app.route('/generate', methods=['POST'])
def generate():
    """Handle generation requests"""
    global model, tokenizer, device
    
    # Get request data
    data = request.json
    prompt = data.get('prompt', '')
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 256)
    
    if not prompt:
        # Check for prompts list for VLLM compatibility
        prompts = data.get('prompts', [])
        if prompts:
            prompt = prompts[0]
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the newly generated text (remove the prompt)
    response_text = generated_text[len(prompt):]
    
    # Format the response like VLLM's response for compatibility
    return jsonify({
        "generated_text": response_text
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

def main():
    parser = argparse.ArgumentParser(description="Run a simple transformers model server")
    parser.add_argument("--model", type=str, required=True, help="Model ID or path")
    parser.add_argument("--hf-token", type=str, help="HuggingFace token for accessing gated models")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run the model on (cuda or cpu)")
    args = parser.parse_args()
    
    global model, tokenizer, device
    device = args.device
    
    print(f"Loading model {args.model}...")
    
    # Set HF token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        print("HuggingFace token set.")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        token=args.hf_token,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    
    print(f"Model loaded successfully. Running on {device}.")
    
    # Start the Flask server
    print(f"Starting server on {args.host}:{args.port}...")
    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main() 