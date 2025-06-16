from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import json

image_dir = 'path/to/LES-AV/images'
label_dir = 'path/to/LES-AV/veins_new'
output_dir = 'path/to/LES-AV/vein_2'

os.makedirs(output_dir, exist_ok=True)
print(f"Output directory exists: {os.path.exists(output_dir)}")
print(f"Output directory writable: {os.access(output_dir, os.W_OK)}")

output_filenames = []
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, image_file)

    if not os.path.exists(label_path):
        continue

    try:
        image = Image.open(image_path)
        label_image = Image.open(label_path).convert('L')
    except Exception:
        continue

    image_array = np.array(image)
    label_array = np.array(label_image)

    if image_array.shape[:2] != label_array.shape:
        continue

    vessel_mask = (label_array == 255).astype(np.float32)
    vessel_mask_expanded = np.repeat(np.expand_dims(vessel_mask, axis=-1), 3, axis=-1)
    inverted_mask_expanded = 1 - vessel_mask_expanded

    vessel_color = np.array([255, 255, 255], dtype=np.uint8)
    vessel_color_expanded = np.full_like(image_array, vessel_color)

    result_image = image_array * inverted_mask_expanded + vessel_color_expanded * vessel_mask_expanded
    result_image = result_image.astype(np.uint8)

    output_filename = f"veins_{image_file}"
    output_path = os.path.join(output_dir, output_filename)

    success = cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    if success:
        output_filenames.append(output_filename)
    else:
        pil_image = Image.fromarray(result_image)
        pil_image.save(output_path)
        output_filenames.append(output_filename)

csv_path = 'path/to/LES-AV/output_filenames2.csv'
df = pd.DataFrame(output_filenames, columns=["Filename"])
df.to_csv(csv_path, index=False)
print(f"Saved filenames to {csv_path}")

full_options = ["arteries", "veins", "microvascular", "optic disk", "optic cup"]

def generate_questions(csv_file):
    df = pd.read_csv(csv_file)
    result = []
    for index, row in df.iterrows():
        new_id = row['new_id']
        correct_answer = row['class']
        if correct_answer not in full_options:
            continue
        wrong_options = [opt for opt in full_options if opt != correct_answer]
        selected_wrong = random.sample(wrong_options, 4)
        all_options = selected_wrong + [correct_answer]
        random.shuffle(all_options)
        letters = list("ABCDE")
        question = "Identify which category the white-labeled structures in the fundus image belong to?\n"
        for letter, opt in zip(letters, all_options):
            question += f"{letter}. {opt}\n"
        idx = all_options.index(correct_answer)
        correct_letter = letters[idx]
        q_dict = {
            "id": new_id,
            "question": question.strip(),
            "answer": correct_letter
        }
        result.append(q_dict)
    return result

if __name__ == "__main__":
    csv_file = 'path/to/LES-AV/LES-AV.csv'
    output_json = 'path/to/LES-AV/LES-AV.json'
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    result = generate_questions(csv_file)
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Generated questions saved to '{output_json}'")
