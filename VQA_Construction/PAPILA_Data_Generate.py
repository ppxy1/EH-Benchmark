import pandas as pd
import json
import random

options = [
    "Normal",
    "Mild Glaucoma",
    "Moderate Glaucoma",
    "Severe Glaucoma"
]

df = pd.read_excel("path/to/Questions.xlsx")
result = []

for index, row in df.iterrows():
    id_value = row['ID']
    question_text = row['Question']
    correct_answer = row['Result']
    if correct_answer not in options:
        continue
    wrong_options = [opt for opt in options if opt != correct_answer]
    selected_wrong = random.sample(wrong_options, 3)
    all_options = selected_wrong + [correct_answer]
    random.shuffle(all_options)
    letters = list("ABCD")
    question_str = f"{question_text}\n"
    for letter, opt in zip(letters, all_options):
        question_str += f"{letter}. {opt}\n"
    correct_idx = all_options.index(correct_answer)
    correct_letter = letters[correct_idx]
    q_dict = {
        "id": id_value,
        "question": question_str.strip(),
        "answer": correct_letter
    }
    result.append(q_dict)

with open("path/to/PAPILA.json", "w") as f:
    json.dump(result, f, indent=4)

print("Generated questions saved to 'PAPILA.json'")
