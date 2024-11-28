import pandas as pd

rocstories_path = "ROCStories.csv"
context_short_stories_path = "Context_Short_Stories.csv"

rocstories_df = pd.read_csv(rocstories_path)
context_short_stories_df = pd.read_csv(context_short_stories_path)

processed_rocstories = []
for _, row in rocstories_df.iterrows():
    input_sentence = row['sentence1'].strip()
    context = ' '.join(input_sentence.split()[:7])
    output = '. '.join([row['sentence2'], row['sentence3'], row['sentence4'], row['sentence5']]).strip()
    processed_rocstories.append({'context': context, 'input': input_sentence, 'output': output})

processed_rocstories_df = pd.DataFrame(processed_rocstories)

combined_df = pd.concat([context_short_stories_df, processed_rocstories_df], ignore_index=True)

combined_dataset_path = "Combined_Stories_Dataset.csv"
combined_df.to_csv(combined_dataset_path, index=False)

print(f"Combined dataset saved to {combined_dataset_path}")
