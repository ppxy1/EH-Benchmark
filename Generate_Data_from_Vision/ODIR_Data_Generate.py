import pandas as pd
import random
import json

full_options = [
    "cataract",
    "normal fundus",
    "laser spot",
    "moderate non proliferative retinopathy",
    "branch retinal artery occlusion",
    "macular epiretinal membrane",
    "mild nonproliferative retinopathy",
    "drusen",
    "vitreous degeneration",
    "retinal pigmentation",
    "pathological myopia",
    "rhegmatogenous retinal detachment",
    "hypertensive retinopathy",
    "diabetic retinopathy",
    "wet age-related macular degeneration",
    "dry age-related macular degeneration",
    "central retinal artery occlusion",
    "retinitis pigmentosa",
    "macular hole",
    "retinochoroidal coloboma",
    "optic disc edema"
]

def generate_questions(excel_file):
    df = pd.read_excel(excel_file)

    result = []

    for index, row in df.iterrows():
        new_id = row['new_id']
        keywords = row['keywords']

        if keywords in full_options:
            wrong_options = [opt for opt in full_options if opt != keywords]
        else:
            wrong_options = full_options.copy()

        selected_wrong = random.sample(wrong_options, 4)

        all_options = selected_wrong + [keywords]

        all_options = list(set(all_options))

        while len(all_options) < 5:
            remaining_wrong = [opt for opt in wrong_options if opt not in all_options]
            all_options.append(random.choice(remaining_wrong))

        random.shuffle(all_options)

        letters = list("ABCDE")

        question = "What we can observe through this fundus?\n"
        for letter, opt in zip(letters, all_options):
            question += f"{letter}. {opt}\n"

        idx = all_options.index(keywords)
        correct_letter = letters[idx]

        q_dict = {
            "id": new_id,
            "question": question,
            "answer": correct_letter
        }

        result.append(q_dict)
    return result

if __name__ == "__main__":
    excel_file = 'new_data.xlsx'
    result = generate_questions(excel_file)

    with open('odir5k.json', 'w') as f:
        json.dump(result, f, indent=4)
