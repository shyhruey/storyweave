from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from safetensors.torch import load_file
import torch

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/generate": {"origins": "http://localhost:3000"}})  # Allow requests from localhost:3000

# Load the model and tokenizer
base_model_name = "gpt2"  # Adjust if your base model is different
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Set up the LoRA adapter configuration
adapter_config = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=["c_attn"]
)

# Apply LoRA adapter to the base model
model = PeftModel(base_model, adapter_config)

# Load adapter weights
adapter_weights_path = './adapter_model.safetensors'  # Ensure this path is correct
adapter_weights = load_file(adapter_weights_path)
model.load_state_dict(adapter_weights, strict=False)
model.eval()
output = ""


@app.route('/generate', methods=['POST'])
def generate_response():
    print("output: ", output)
    data = request.json
    user_input = data.get("input", "")
    user_input = output + " " + user_input
    # Generate response from the model
    with torch.no_grad():
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=1.0,
            top_p=0.85,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            do_sample=True
        )
    
    # Decode the model's output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the output if needed (e.g., removing repeated input)
    if generated_text.lower().startswith(user_input.lower()):
        generated_text = generated_text[len(user_input):].strip()
    
    # Limit to the first sentence and 20 words
    sentences = generated_text.split(". ")
    first_sentence = sentences[0] + "." if sentences else generated_text
    words = first_sentence.split()
    if len(words) > 20:
        first_sentence = " ".join(words[:20]) + "..."
    output = first_sentence
    return jsonify({"response": first_sentence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)