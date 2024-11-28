# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel, LoraConfig
# from safetensors.torch import load_file
# import torch

# # Load base model and tokenizer
# base_model_name = "gpt2"  # Adjust if your base model is different
# tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# # Set up the LoRA adapter configuration inline
# adapter_config = LoraConfig(
#     r=4,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     task_type="CAUSAL_LM",
#     target_modules=["c_attn"]
# )

# # Apply LoRA adapter to the base model
# model = PeftModel(base_model, adapter_config)

# # Load the adapter weights with safetensors
# adapter_weights_path = './adapter_model.safetensors'  # Ensure this path is correct
# adapter_weights = load_file(adapter_weights_path)     # Load safetensors weights

# # Load weights into the model
# model.load_state_dict(adapter_weights, strict=False)
# model.eval()

# # Example text generation
# input_text = "Once upon a time,"
# inputs = tokenizer(input_text, return_tensors="pt")
# with torch.no_grad():
#     outputs = model.generate(**inputs, max_length=50)

# # Decode and print the generated text
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))



from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from safetensors.torch import load_file
import torch
from random import choice 

# Load base model and tokenizer
base_model_name = "gpt2"  # Adjust if your base model is different
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Set up the LoRA adapter configuration inline
adapter_config = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=["c_attn"]
)

# Apply LoRA adapter to the base model
model = PeftModel(base_model, adapter_config)

# Load the adapter weights with safetensors
adapter_weights_path = './adapter_model.safetensors'  # Ensure this path is correct
adapter_weights = load_file(adapter_weights_path)     # Load safetensors weights

# Load weights into the model
model.load_state_dict(adapter_weights, strict=False)
model.eval()
output = ""
# Initial story prompt by the model
print("STORY START: Once upon a time\n")

# Interactive storytelling loop
print("Enter your input to continue the story, or type 'END' to stop.")
while True:
    # Take user input
    user_input = input("You: ")
    if user_input.strip().upper() == "END":
        print("Storytelling session ended.")
        break
    ## add context
    user_input = output + " " + user_input
    # Generate model's response based solely on the user's input
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

    # Decode the response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Step 1: Remove repeated user input if present at the beginning
    if generated_text.lower().startswith(user_input.lower()):
        generated_text = generated_text[len(user_input):].strip()

    # Step 2: Limit output to the first sentence or a set number of words
    sentences = generated_text.split(". ")
    first_sentence = sentences[0] + "." if sentences else generated_text  # Keep only the first sentence

    # Limit output length to 20 words to avoid overly long responses
    words = first_sentence.split()
    if len(words) > 20:
        first_sentence = " ".join(words[:20]) + "..."
    output = first_sentence
    # Print the model's continuation
    print(f"AI: {first_sentence}\n")