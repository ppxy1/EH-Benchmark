import os
import pandas as pd
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

image_folder = 'path/to/Images'
json_folder = 'path/to/json'
output_folder = 'path/to/output/cup'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def plot_cup_labels(image_folder, json_folder, output_folder):
    counter = 1
    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            json_name = os.path.splitext(image_name)[0] + '.json'
            json_path = os.path.join(json_folder, json_name)
            image_path = os.path.join(image_folder, image_name)
            if os.path.exists(json_path):
                img = Image.open(image_path)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                for shape in data['shapes']:
                    if shape['label'] == 'cup':
                        points = shape['points']
                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.imshow(img)
                        polygon = patches.Polygon(points, closed=True, edgecolor='white', linewidth=4, fill=False)
                        ax.add_patch(polygon)
                        ax.axis('off')
                        plt.tight_layout()
                        output_image_path = os.path.join(output_folder, f'cup_{counter}.png')
                        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.0, transparent=False)
                        plt.close(fig)
                        img_resized = Image.open(output_image_path)
                        img_resized = img_resized.resize((512, 512), Image.Resampling.LANCZOS)
                        img_resized.save(output_image_path)
                        counter += 1
                        break

plot_cup_labels(image_folder, json_folder, output_folder)

image_folder = 'path/to/location/G1020/disc'
image_data = []
for image_name in os.listdir(image_folder):
    if image_name.endswith('.jpg') or image_name.endswith('.png'):
        image_path = f'images/{image_name}'
        image_data.append({'new_id': image_path})

df = pd.DataFrame(image_data)
output_csv_path = 'path/to/location/G1020/G1020.csv'
df.to_csv(output_csv_path, index=False)

print(f"CSV file has been saved to {output_csv_path}")

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
    csv_file = 'path/to/location/G1020/G1020.csv'
    output_json = 'path/to/location/G1020/G1020.json'
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    result = generate_questions(csv_file)
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Generated questions saved to '{output_json}'")
