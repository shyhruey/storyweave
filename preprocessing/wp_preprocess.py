import pandas as pd
import re

with open("writingPrompts/valid.wp_source", "r", encoding="utf-8") as f_source, \
     open("writingPrompts/valid.wp_target", "r", encoding="utf-8") as f_target:
    
    contexts = [line.strip() for line in f_source]
    full_outputs = [line.strip() for line in f_target]

contexts = [re.sub(r'\[.*?\]', '', line).strip() for line in contexts]

inputs = [" ".join(line.split()[:6]) for line in full_outputs]

full_outputs = [line.replace("<newline>", "").strip() for line in full_outputs]

df = pd.DataFrame({
    "context": contexts,
    "input": inputs,
    "output": full_outputs
})

df.to_csv("writing_prompts_custom.csv", index=False, encoding="utf-8")
print("Converted to writing_prompts_custom.csv with context, input, and output columns")
