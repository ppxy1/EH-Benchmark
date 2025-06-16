import os
import shutil
import pandas as pd
import random
import json

def move_images_and_create_csv(mh_folder, retina_folder, choroid_folder, output_image_folder, csv_path):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    data = []
    valid_extensions = ('.png', '.jpg', '.jpeg')

    if os.path.exists(mh_folder):
        for filename in os.listdir(mh_folder):
            if filename.lower().endswith(valid_extensions):
                src_path = os.path.join(mh_folder, filename)
                dst_path = os.path.join(output_image_folder, filename)
                try:
                    shutil.move(src_path, dst_path)
                    data.append({
                        'new_id': f"images/{filename}",
                        'class': 'Macular Hole'
                    })
                except Exception:
                    continue

    if os.path.exists(retina_folder):
        for filename in os.listdir(retina_folder):
            if filename.lower().endswith(valid_extensions):
                src_path = os.path.join(retina_folder, filename)
                dst_path = os.path.join(output_image_folder, filename)
                try:
                    shutil.move(src_path, dst_path)
                    data.append({
                        'new_id': f"images/{filename}",
                        'class': 'Retina'
                    })
                except Exception:
                    continue

    if os.path.exists(choroid_folder):
        for filename in os.listdir(choroid_folder):
            if filename.lower().endswith(valid_extensions):
                src_path = os.path.join(choroid_folder, filename)
                dst_path = os.path.join(output_image_folder, filename)
                try:
                    shutil.move(src_path, dst_path)
                    data.append({
                        'new_id': f"images/{filename}",
                        'class': 'Choroid'
                    })
                except Exception:
                    continue

    if not data:
        return

    df = pd.DataFrame(data, columns=['new_id', 'class'])
    try:
        df.to_csv(csv_path, index=False)
    except Exception:
        pass

if __name__ == "__main__":
    mh_folder = "path/to/mh_mask"
    retina_folder = "path/to/retina_mask"
    choroid_folder = "path/to/choroid_mask"
    output_image_folder = "path/to/images"
    csv_path = "path/to/OIMHS.csv"
    move_images_and_create_csv(mh_folder, retina_folder, choroid_folder, output_image_folder, csv_path)

full_options = [
    'Choroid',
    'Intraretinal Cyst',
    'Macular Hole',
    'Retina',
    'Internal Limiting Membrane',
    'Pigment Epithelial Detachment',
    'Choroidal Neovascularization'
]

def generate_questions(csv_file):
    df = pd.read_csv(csv_file)
    result = []
    for index, row in df.iterrows():
        new_id = row['new_id']
        correct_class = row['class']
        if correct_class not in full_options:
            continue
        wrong_options = [opt for opt in full_options if opt != correct_class]
        selected_wrong = random.sample(wrong_options, 4)
        all_options = selected_wrong + [correct_class]
        random.shuffle(all_options)
        letters = list("ABCDE")
        question = "In OCT, what type of lesion is marked with a red bounding box?\n"
        for letter, opt in zip(letters, all_options):
            question += f"{letter}. {opt}\n"
        correct_idx = all_options.index(correct_class)
        correct_letter = letters[correct_idx]
        q_dict = {
            "id": new_id,
            "question": question.strip(),
            "answer": correct_letter
        }
        result.append(q_dict)
    return result

if __name__ == "__main__":
    csv_file = "path/to/OIMHS.csv"
    output_json = "path/to/OIMHS.json"
    result = generate_questions(csv_file)
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=4)
