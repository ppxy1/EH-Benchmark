import pandas as pd
import os
from PIL import Image
import numpy as np
import random
import json

df = pd.read_csv('path/to/IDRID.csv')
mask_folder = 'path/to/masks/4. Soft Exudates'
mask_files = os.listdir(mask_folder)

for mask_file in mask_files:
    base_name = os.path.splitext(mask_file)[0]
    core_name = base_name.replace('_SE', '')
    matching_row = df[df['new_id'].apply(lambda x: os.path.splitext(os.path.basename(x))[0]) == core_name]
    if matching_row.empty:
        continue
    index = matching_row.index[0]
    position = matching_row['position'].values[0]
    mask_path = os.path.join(mask_folder, mask_file)
    mask_image = Image.open(mask_path)
    if mask_image.mode == 'RGB':
        mask_array = np.array(mask_image)
        red_channel = mask_array[:, :, 0]
        mask_binary = (red_channel > 0)
    else:
        mask_image = mask_image.convert('L')
        mask_array = np.array(mask_image)
        mask_binary = (mask_array == 255)
    height, width = mask_binary.shape
    cy = height // 2
    cx = width // 2
    superior_mask = np.zeros_like(mask_binary, dtype=bool)
    superior_mask[:cy, :] = True
    inferior_mask = np.zeros_like(mask_binary, dtype=bool)
    inferior_mask[cy:, :] = True
    if position == 'OD':
        nasal_mask = np.zeros_like(mask_binary, dtype=bool)
        nasal_mask[:, cx:] = True
        temporal_mask = np.zeros_like(mask_binary, dtype=bool)
        temporal_mask[:, :cx] = True
    else:
        nasal_mask = np.zeros_like(mask_binary, dtype=bool)
        nasal_mask[:, :cx] = True
        temporal_mask = np.zeros_like(mask_binary, dtype=bool)
        temporal_mask[:, cx:] = True
    regions = []
    if np.any(mask_binary & superior_mask):
        regions.append('Superior')
    if np.any(mask_binary & inferior_mask):
        regions.append('Inferior')
    if np.any(mask_binary & nasal_mask):
        regions.append('Nasal')
    if np.any(mask_binary & temporal_mask):
        regions.append('Temporal')
    if not regions:
        df.at[index, 'Soft Exudate'] = 'None'
    else:
        df.at[index, 'Soft Exudate'] = ', '.join(sorted(regions))

df.to_csv('path/to/IDRID.csv', index=False)

df = pd.read_csv('path/to/IDRID.csv')
print(df['Soft Exudate'].unique())

full_options = [
    'Inferior, Nasal, Superior, Temporal',
    'Inferior, Superior, Temporal',
    'Inferior, Nasal, Superior',
    'Nasal, Superior, Temporal',
    'Superior, Temporal',
    'Nasal, Superior',
    'Inferior, Temporal',
    'Inferior, Nasal, Temporal',
    'None'
]

def generate_questions(csv_file):
    df = pd.read_csv(csv_file)
    df = df[df['Soft Exudate'].notna() & ~df['Soft Exudate'].isin(['Mask not found', 'Mask not loaded'])]
    result = []
    for index, row in df.iterrows():
        new_id = row['new_id']
        location = row['Soft Exudate']
        if location == 'None':
            continue
        if location not in full_options:
            continue
        wrong_options = [opt for opt in full_options if opt != location]
        selected_wrong = random.sample(wrong_options, 4)
        all_options = selected_wrong + [location]
        random.shuffle(all_options)
        letters = list("ABCDE")
        question = "Where does Soft Exudate occur?\n"
        for letter, opt in zip(letters, all_options):
            question += f"{letter}. {opt}\n"
        idx = all_options.index(location)
        correct_letter = letters[idx]
        q_dict = {
            "id": f"images/{new_id}",
            "question": question.strip(),
            "answer": correct_letter
        }
        result.append(q_dict)
    return result

if __name__ == "__main__":
    csv_file = 'path/to/IDRID.csv'
    result = generate_questions(csv_file)
    with open('path/to/Soft_Exudate.json', 'w') as f:
        json.dump(result, f, indent=4)
    print("Generated questions saved to 'Soft_Exudate.json'")
