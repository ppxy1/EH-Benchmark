import os
import shutil
import pandas as pd
from pathlib import Path
import random
import json

category_mapping = {
    "VID": "Vitreomacular Interface Disease",
    "RVO": "Retinal Vein Occlusion",
    "NO": "Normal",
    "RAO": "Retinal Artery Occlusion",
    "ERM": "Epiretinal Membrane",
    "DME": "Diabetic Macular Edema",
    "AMD": "Age-related Macular Degeneration"
}

source_dir = "path/to/OCTDL"
target_dir = "path/to/images"
Path(target_dir).mkdir(exist_ok=True)

data_list = []

for category in category_mapping.keys():
    category_path = os.path.join(source_dir, category)
    if not os.path.exists(category_path):
        continue
    keywords = category_mapping[category]
    for filename in os.listdir(category_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            source_file = os.path.join(category_path, filename)
            target_file = os.path.join(target_dir, filename)
            shutil.move(source_file, target_file)
            new_id = f"images/{filename}"
            data_list.append({
                "new_id": new_id,
                "keywords": keywords
            })

df = pd.DataFrame(data_list)
output_csv = "path/to/OCTDL/octdl_data.csv"
df.to_csv(output_csv, index=False)
print(f"Total images processed: {len(data_list)}")

full_options = [
    "Vitreomacular Interface Disease",
    "Retinal Vein Occlusion",
    "Normal",
    "Retinal Artery Occlusion",
    "Epiretinal Membrane",
    "Diabetic Macular Edema",
    "Age-related Macular Degeneration"
]

def generate_questions(csv_file):
    df = pd.read_csv(csv_file)
    result = []
    for index, row in df.iterrows():
        new_id = row['new_id']
        keywords = row['keywords']
        wrong_options = [opt for opt in full_options if opt != keywords]
        selected_wrong = random.sample(wrong_options, 4)
        all_options = selected_wrong + [keywords]
        all_options = list(set(all_options))
        while len(all_options) < 5:
            remaining_wrong = [opt for opt in wrong_options if opt not in all_options]
            all_options.append(random.choice(remaining_wrong))
        random.shuffle(all_options)
        letters = list("ABCDE")
        question = "What we can observe through this OCT?\n"
        for letter, opt in zip(letters, all_options):
            question += f"{letter}. {opt}\n"
        idx = all_options.index(keywords)
        correct_letter = letters[idx]
        q_dict = {
            "id": new_id,
            "question": question.strip(),
            "answer": correct_letter
        }
        result.append(q_dict)
    return result

if __name__ == "__main__":
    csv_file = "path/to/OCTDL/octdl_data.csv"
    result = generate_questions(csv_file)
    with open("path/to/OCTDL/octdl.json", 'w') as f:
        json.dump(result, f, indent=4)
    print("Generated questions saved to 'octdl.json'")
