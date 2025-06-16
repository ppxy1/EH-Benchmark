import pandas as pd
import json

options = [
    "DR0: No Diabetic Retinopathy",
    "DR1: Mild Non-Proliferative Diabetic Retinopathy",
    "DR2: Moderate Non-Proliferative Diabetic Retinopathy",
    "DR3: Severe Non-Proliferative Diabetic Retinopathy",
    "DR4: Proliferative Diabetic Retinopathy"
]

df = pd.read_csv('retinal-lesions/dr_grades.csv')
result = []

for index, row in df.iterrows():
    image_id = row['image id']
    class_label = row['class']

    if class_label not in options:
        continue

    new_id = f"images/{image_id}.jpg"
    question = "What is the diabetic retinopathy grade of the image?\n"
    letters = ['A', 'B', 'C', 'D', 'E']
    for letter, opt in zip(letters, options):
        question += f"{letter}. {opt}\n"

    idx = options.index(class_label)
    correct_letter = letters[idx]

    entry = {
        "id": new_id,
        "question": question.strip(),
        "answer": correct_letter
    }

    result.append(entry)

with open('retinal-lesions/DR_Grade.json', 'w') as f:
    json.dump(result, f, indent=4)