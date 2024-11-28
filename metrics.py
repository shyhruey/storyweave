from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import language_tool_python
from sentence_transformers import SentenceTransformer, util

tool = language_tool_python.LanguageTool('en-US')
coherence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def generate_story(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=500)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=1.0,
            top_p=0.85,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            do_sample=True
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text.strip()


def calculate_grammar_errors(story):
    matches = tool.check(story)
    error_count = len(matches)
    total_sentences = 1

    error_rate = error_count / total_sentences if total_sentences > 0 else 0
    return error_rate

def calculate_semantic_coherence(prompt, story):
    prompt_embedding = coherence_model.encode(prompt, convert_to_tensor=True)
    story_embedding = coherence_model.encode(story, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(prompt_embedding, story_embedding).item()
    return similarity


test_prompts = [
    "In a land of dragons and knights, a hidden kingdom awaited discovery.",
    "John's alarm didn’t go off, and now he’s late for work.",
    "The old clock in the attic began to tick after years of silence.",
    "You stand before three doors: one red, one blue, and one green. Which one do you choose?",
    "I don't like any of these options; I want to fly away instead.",
    "It was 1942, and the world was at war. Amid the chaos, a secret message arrived.",
    "The spaceship’s lights flickered as an unknown lifeform approached the crew.",
    "The colors swirled together, forming shapes that whispered secrets to the artist.",
    "A storm.",
    "The ship sailed across the ocean, guided by a lone captain who held a map leading to a treasure hidden by ancient pirates, while a rival crew followed close behind.",
    "It was a bright sunny day, but the darkness was overwhelming.",
    "She stared at the letter in her hands, tears streaming down her face.",
    "A bunny found a magic carrot in the middle of the forest.",
    "The superhero tripped on his cape and spilled coffee on his nemesis during a peace summit.",
    "Once upon a time in a castle filled with shadows and secrets."
]

tokenizer1 = AutoTokenizer.from_pretrained("gpt2")
base_model1 = AutoModelForCausalLM.from_pretrained("gpt2")
model1 = PeftModel.from_pretrained(base_model1, "./model1")

tokenizer2 = AutoTokenizer.from_pretrained("gpt2")
base_model2 = AutoModelForCausalLM.from_pretrained("gpt2")
model2 = PeftModel.from_pretrained(base_model2, "./model2")

tokenizer3 = AutoTokenizer.from_pretrained("gpt2")
base_model3 = AutoModelForCausalLM.from_pretrained("gpt2")
model3 = PeftModel.from_pretrained(base_model2, "./model3")


def evaluate_models(test_prompts, model, tokenizer):
    total_grammar_error_rate = 0
    total_semantic_coherence = 0
    num_prompts = len(test_prompts)

    for prompt in test_prompts:
        story = generate_story(model, tokenizer, prompt)

        grammar_error_rate = calculate_grammar_errors(story)
        total_grammar_error_rate += grammar_error_rate

        semantic_coherence = calculate_semantic_coherence(prompt, story)
        total_semantic_coherence += semantic_coherence

        print(f"\nPrompt: {prompt}")
        print(f"Generated Story: {story}")
        print(f"Grammar Error Rate: {grammar_error_rate:.2f}")
        print(f"Semantic Coherence: {semantic_coherence:.2f}")

    avg_grammar_error_rate = total_grammar_error_rate / num_prompts
    avg_semantic_coherence = total_semantic_coherence / num_prompts
    return avg_grammar_error_rate, avg_semantic_coherence


print("\nEvaluating Model 1:")
grammar_error_rate1, semantic_coherence1 = evaluate_models(test_prompts, model1, tokenizer1)
print("\nEvaluating Model 2:")
grammar_error_rate2, semantic_coherence2 = evaluate_models(test_prompts, model2, tokenizer2)
print("\nEvaluating Model 3:")
grammar_error_rate3, semantic_coherence3 = evaluate_models(test_prompts, model3, tokenizer3)


print(f"\nModel 1 - Average Grammar Error Rate: {grammar_error_rate1:.2f}, Average Semantic Coherence: {semantic_coherence1:.2f}")
print(f"Model 2 - Average Grammar Error Rate: {grammar_error_rate2:.2f}, Average Semantic Coherence: {semantic_coherence2:.2f}")
print(f"Model 3 - Average Grammar Error Rate: {grammar_error_rate2:.2f}, Average Semantic Coherence: {semantic_coherence2:.2f}")