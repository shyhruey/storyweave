from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-small"  # replace with our model
tokeniser = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_story_segment(context, user_choice=None):
    """
    generate the next part of story with user choices
    
    input:
        context (str): summary of the story so far
        user_choice (str, optional): latest user choice
    
    output:
        tuple: generated story and a list of choices
    """
    # create prompt for generating story continuation with choices
    prompt = f"Context: {context}\n"
    if user_choice:
        prompt += f"User choice: {user_choice}\n"
    prompt += "Continue the story and suggest two choices for the user:"

    input_ids = tokeniser(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=150, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    generated_text = tokeniser.decode(outputs[0], skip_special_tokens=True)
    
    # split the generated text into story and choices based on format
    if "Choices:" in generated_text:
        story_part, choices_part = generated_text.split("Choices:", 1)
        choices = [choice.strip() for choice in choices_part.split("\n") if choice.strip()]
    else:
        story_part = generated_text
        choices = ["Continue", "End the story"]

    return story_part.strip(), choices
