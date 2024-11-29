import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if device == "cuda" else ""

# Load and process the data, dropping unnecessary index column
df = pd.read_csv("better_processed_data2.0.csv").drop(['Unnamed: 0'], axis=1)

# Verify columns
required_columns = {'context', 'input', 'output'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"The CSV file is missing one or more required columns: {required_columns}")

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load GPT-2 model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    load_in_8bit=True if device == "cuda" else False,
    device_map='auto' if device == "cuda" else None
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Post-process the model for LoRA fine-tuning
for param in model.parameters():
    param.requires_grad = False  # Freeze base model parameters
    if param.ndim == 1:
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

# Define LoRA configuration
config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

# Print the number of trainable parameters
def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params} || Total params: {total_params} || Trainable %: {100 * trainable_params / total_params:.2f}%")

print_trainable_parameters(model)

# Define function to generate the prompt
def generate_prompt(context: str, input: str, output: str) -> str:
    return f"### Context:\n{context}\n\n### Input:\n{input}\n\n### Output:\n{output}</s>"

# Tokenization function
def tokenize_function(samples):
    # Generate prompt for each row
    prompts = [
        generate_prompt(context, input_text, output)
        for context, input_text, output in zip(samples['context'], samples['input'], samples['output'])
    ]
    # Tokenize with dynamic padding to the longest sequence in the batch
    return tokenizer(
        prompts,
        truncation=True,
        padding="longest",  # Adjust to the longest sequence in the batch
        return_tensors="pt"  # Ensure tensors match model expectations
    )

# Apply tokenization with batched mapping
mapped_dataset = dataset.map(tokenize_function, batched=True)
# Set up TrainingArguments
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=3,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=100,
    learning_rate=1e-3,
    fp16=True if device == "cuda" else False,
    logging_steps=1,
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=1000)

# Initialize Trainer
trainer = Trainer(
    model=model,
    train_dataset=mapped_dataset,
    args=training_args,
    data_collator=data_collator,
)

# Disable cache during training for memory efficiency
model.config.use_cache = False
trainer.train()

# Save the fine-tuned model
trainer.save_model("fine_tuned_gpt2_lora")


# Example usage
# context = "in the darkest night"
# input_text = "I decided to"
# generated_text = make_inference(context, input_text)
# print("Generated Story:", generated_text)


### TRY WITH ENTITIY
# import spacy

# # Load the pre-trained small English model
# nlp = spacy.load("en_core_web_sm")

# context = input('Enter a starting prompt: ')
# input_text = input('Enter an action ')
# generated_text = make_inference(context, input_text)
# print(generated_text)
# output = ((generated_text.split('###')[3]).split(':')[1]).lstrip('\n').split('.')[0]
# input_text = input('Enter an action: ')
# while (input_text != 'quit()'):
#   try:
#       # sentences = output.split('.')[:2]
#       #   # Process each sentence and extract entities
#       # for sentence in sentences:
#       #   doc = nlp(sentence)
#       #   for ent in doc.ents:
#       #     print(f"Entity: {ent.text}, Label: {ent.label_}")
#       #     input_text = input_text + ' ' + ent.text
#       # print("new input: ", input_text)
#       generated_text = make_inference(output, input_text)
#       output = ((generated_text.split('###')[3]).split(':')[1]).lstrip('\n').split('.')[0]
#       print(generated_text)

#   except Exception as e:
#       print('No output')
#   finally:
#       input_text = input('Enter an action: ')