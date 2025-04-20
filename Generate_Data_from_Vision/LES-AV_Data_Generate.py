from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd

import random
import json

# image_dir = 'images'
# label_dir = 'LES-AV/veins_new'
# output_dir = 'LES-AV/vein_2'
#
# os.makedirs(output_dir, exist_ok=True)
# print(f"Output directory exists: {os.path.exists(output_dir)}")
# print(f"Output directory writable: {os.access(output_dir, os.W_OK)}")
#
# output_filenames = []
#
# image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
#
# for image_file in image_files:
#
#     image_path = os.path.join(image_dir, image_file)
#     label_path = os.path.join(label_dir, image_file)
#
#     if not os.path.exists(label_path):
#         print(f"Label file not found for {image_file}, skipping...")
#         continue
#
#     try:
#         image = Image.open(image_path)
#         label_image = Image.open(label_path).convert('L')
#     except Exception as e:
#         print(f"Error loading files for {image_file}: {e}")
#         continue
#
#     image_array = np.array(image)
#     label_array = np.array(label_image)
#
#     if image_array.shape[:2] != label_array.shape:
#         print(f"Dimension mismatch for {image_file}, skipping...")
#         continue
#
#     vessel_mask = (label_array == 255).astype(np.float32)
#     vessel_mask_expanded = np.repeat(np.expand_dims(vessel_mask, axis=-1), 3, axis=-1)
#     inverted_mask_expanded = 1 - vessel_mask_expanded
#
#     vessel_color = np.array([255, 255, 255], dtype=np.uint8)
#     vessel_color_expanded = np.full_like(image_array, vessel_color)
#
#     result_image = image_array * inverted_mask_expanded + vessel_color_expanded * vessel_mask_expanded
#     result_image = result_image.astype(np.uint8)
#
#     output_filename = f"veins_{image_file}"
#     output_path = os.path.join(output_dir, output_filename)
#
#     success = cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
#     if success:
#         print(f"Saved {output_filename}")
#         output_filenames.append(output_filename)
#     else:
#         print(f"Failed to save {output_filename}, trying PIL...")
#         pil_image = Image.fromarray(result_image)
#         pil_image.save(output_path)
#         print(f"PIL saved {output_filename}")
#         output_filenames.append(output_filename)
#
# csv_path = 'output_filenames2.csv'
# df = pd.DataFrame(output_filenames, columns=["Filename"])
# df.to_csv(csv_path, index=False)
# print(f"Saved filenames to {csv_path}")

# 定义所有可能的选项
full_options = ["arteries", "veins", "microvascular", "optic disk", "optic cup"]

def generate_questions(csv_file):
    # 加载 CSV 文件
    df = pd.read_csv(csv_file)

    result = []
    for index, row in df.iterrows():
        new_id = row['new_id']
        correct_answer = row['class']

        # 确保正确答案在选项列表中
        if correct_answer not in full_options:
            print(f"Invalid class '{correct_answer}' for {new_id}, skipping...")
            continue

        # 生成错误选项：从 full_options 中排除正确答案后随机选择 4 个
        wrong_options = [opt for opt in full_options if opt != correct_answer]
        selected_wrong = random.sample(wrong_options, 4)

        # 组合所有选项：4 个错误选项 + 1 个正确答案
        all_options = selected_wrong + [correct_answer]
        random.shuffle(all_options)  # 随机打乱选项顺序

        # 生成问题字符串
        letters = list("ABCDE")
        question = "Identify which category the white-labeled structures in the fundus image belong to?\n"
        for letter, opt in zip(letters, all_options):
            question += f"{letter}. {opt}\n"

        # 确定正确答案的字母
        idx = all_options.index(correct_answer)
        correct_letter = letters[idx]

        # 构造问题字典
        q_dict = {
            "id": new_id,  # 直接使用 new_id（如 "images/arteries_12.png"）
            "question": question.strip(),
            "answer": correct_letter
        }

        result.append(q_dict)

    return result

if __name__ == "__main__":
    # 路径设置
    csv_file = 'LES-AV.csv'
    output_json = 'LES-AV.json'

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    # 生成问题
    result = generate_questions(csv_file)

    # 保存到 JSON 文件
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Generated questions saved to '{output_json}'")