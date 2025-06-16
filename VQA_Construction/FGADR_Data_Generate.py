import os
import json
import base64
from tqdm import tqdm
import csv
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import random

df = pd.read_csv('path/to/FGADR.csv')
mask_folder = 'path/to/HardExudate_Masks'
output_csv_path = 'path/to/FGADR.csv'

for filename in os.listdir(mask_folder):
    if not filename.lower().endswith(".png"):
        continue
    base_name = os.path.splitext(filename)[0]
    mask_path = os.path.join(mask_folder, filename)
    matched_row = df[df['new_id'].str.contains(base_name)]
    if matched_row.empty:
        continue
    index = matched_row.index[0]
    position = matched_row.iloc[0]['position']
    try:
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)
    except Exception:
        df.at[index, 'HardExudate'] = 'Mask not loaded'
        continue
    mask_binary = mask_np == 255
    h, w = mask_binary.shape
    cx = w // 2
    cy = h // 2
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
        df.at[index, 'HardExudate'] = 'None'
    else:
        df.at[index, 'HardExudate'] = ', '.join(sorted(regions))

df.to_csv(output_csv_path, index=False)

df = pd.read_csv('path/to/FGADR.csv')
print(df['HardExudate'].unique())

full_options = [
    'Inferior, Temporal',
    'Inferior, Superior, Temporal',
    'Inferior, Nasal, Superior, Temporal',
    'Nasal, Superior, Temporal',
    'Nasal, Superior',
    'Inferior, Nasal, Temporal',
    'Superior, Temporal',
    'Inferior, Nasal',
    'Inferior, Nasal, Superior',
    'None'
]

def generate_questions(csv_file):
    df = pd.read_csv(csv_file)
    df = df[df['HardExudate'].notna() & ~df['HardExudate'].isin(['Mask not found', 'Mask not loaded'])]
    result = []
    for index, row in df.iterrows():
        new_id = row['new_id']
        location = row['HardExudate']
        if location == 'None':
            continue
        if location not in full_options:
            continue
        wrong_options = [opt for opt in full_options if opt != location]
        selected_wrong = random.sample(wrong_options, 4)
        all_options = selected_wrong + [location]
        random.shuffle(all_options)
        letters = list("ABCDE")
        question = "Where does Hard exudate occur?\n"
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
    csv_file = 'path/to/FGADR.csv'
    result = generate_questions(csv_file)
    with open('path/to/HardExudate.json', 'w') as f:
        json.dump(result, f, indent=4)
    print("Generated questions saved to 'HardExudate.json'")
