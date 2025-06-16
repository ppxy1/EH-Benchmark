import pandas as pd
import cv2
import os
from natsort import natsorted
import numpy as np
import pathlib
import imageio
from PIL import Image, ImageDraw
import random
import json

df = pd.read_csv('path/to/OCT5k/all_bounding_boxes.csv')
output_folder = 'path/to/OCT5k/images'
os.makedirs(output_folder, exist_ok=True)

grouped = df.groupby(['image', 'class'])
output_data = []

for (image, class_name), group in grouped:
    image_path = f'path/to/OCT5k/Photo/{image}'
    if not os.path.exists(image_path):
        continue
    img_pil = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img_pil)
    for _, row in group.iterrows():
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0), width=2)
    base_name = os.path.basename(image).replace(' ', '_')
    save_name = f"{base_name}_{class_name}.png"
    save_path = os.path.join(output_folder, save_name)
    img_pil.save(save_path)
    output_data.append({
        'image_name': save_name,
        'class': class_name
    })

output_df = pd.DataFrame(output_data)
output_csv_path = 'path/to/OCT5k/OCT5k.csv'
output_df.to_csv(output_csv_path, index=False)

class_mapping = {
    'Geographicatrophy': 'geographic atrophy',
    'Softdrusen': 'soft drusen',
    'Harddrusen': 'hard drusen',
    'Reticulardrusen': 'reticular drusen',
    'PRlayerdisruption': 'photoreceptor layer disruption',
    'SoftdrusenPED': 'soft drusen PED',
    'Choroidalfolds': 'choroidal folds',
    'Hyperfluorescentspots': 'hyperfluorescent spots',
    'Fluid': 'retinal fluid'
}

df['class'] = df['class'].map(class_mapping).fillna(df['class'])
output_path = 'path/to/OCT5k/OCT5k.csv'
df.to_csv(output_path, index=False)

full_options = [
    'geographic atrophy',
    'soft drusen',
    'hard drusen',
    'reticular drusen',
    'photoreceptor layer disruption',
    'soft drusen PED',
    'choroidal folds',
    'hyperfluorescent spots',
    'retinal fluid'
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
    csv_file = 'path/to/OCT5k/OCT5k.csv'
    result = generate_questions(csv_file)
    with open('path/to/OCT5k/OCT5k.json', 'w') as f:
        json.dump(result, f, indent=4)
    print("Generated questions saved to 'OCT5k.json'")

