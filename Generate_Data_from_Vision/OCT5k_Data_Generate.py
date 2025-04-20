import pandas as pd
import cv2
import os
from natsort import natsorted
import numpy as np
import pathlib
import pandas as pd
import imageio
from PIL import Image, ImageDraw

# # 读取原始 CSV 文件
# df = pd.read_csv('all_bounding_boxes.csv')
#
# # 创建输出文件夹
# output_folder = 'OCT5k\Detection\images'
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # 按图像和类别分组
# grouped = df.groupby(['image', 'class'])
#
# # 初始化一个列表，用于记录生成的图像名称和类别
# output_data = []
#
# # 遍历每个分组
# for (image, class_name), group in grouped:
#     image_path = f'OCT5k/Detection/Photo/{image}'
#
#     # 检查图像文件是否存在
#     if not os.path.exists(image_path):
#         print(f"图像未找到: {image_path}")
#         continue
#
#     # 使用 PIL 加载图像
#     img_pil = Image.open(image_path).convert('RGB')
#     draw = ImageDraw.Draw(img_pil)
#
#     # 遍历分组中的每个边界框
#     for _, row in group.iterrows():
#         xmin = int(row['xmin'])
#         ymin = int(row['ymin'])
#         xmax = int(row['xmax'])
#         ymax = int(row['ymax'])
#
#         # 在图像上绘制红色矩形框
#         draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0), width=2)
#
#     # 生成保存文件名，包含图像名和类别名
#     base_name = os.path.basename(image).replace(' ', '_')
#     save_name = f"{base_name}_{class_name}.png"
#     save_path = os.path.join(output_folder, save_name)
#
#     # 保存图像
#     img_pil.save(save_path)
#     print(f"已保存: {save_path}")
#
#     # 记录图像名称和类别
#     output_data.append({
#         'image_name': save_name,
#         'class': class_name
#     })
#
# # 将记录的数据保存为新的 CSV 文件
# output_df = pd.DataFrame(output_data)
# output_csv_path = 'OCT5k.csv'
# output_df.to_csv(output_csv_path, index=False)
# print(f"已将图像名称和类别保存到: {output_csv_path}")

import pandas as pd


# # 创建一个字典，将旧的class值映射为新的class值
# class_mapping = {
#     'Geographicatrophy': 'geographic atrophy',
#     'Softdrusen': 'soft drusen',
#     'Harddrusen': 'hard drusen',
#     'Reticulardrusen': 'reticular drusen',
#     'PRlayerdisruption': 'photoreceptor layer disruption',
#     'SoftdrusenPED': 'soft drusen PED',
#     'Choroidalfolds': 'choroidal folds',
#     'Hyperfluorescentspots': 'hyperfluorescent spots',
#     'Fluid': 'retinal fluid'
# }
#
# # 对'class'列进行映射转换
# df['class'] = df['class'].map(class_mapping).fillna(df['class'])
#
# # 保存修改后的 CSV 文件
# output_path = 'OCT5k\OCT5k.csv'
# df.to_csv(output_path, index=False)
#
# print(f"Modified CSV file saved to: {output_path}")

import pandas as pd
import random
import json

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
    # Read the CSV file
    df = pd.read_csv(csv_file)
    result = []

    for index, row in df.iterrows():
        new_id = row['new_id']
        correct_class = row['class']

        if correct_class not in full_options:
            print(f"Warning: {correct_class} not in full_options, skipping this entry")
            continue

        wrong_options = [opt for opt in full_options if opt != correct_class]
        selected_wrong = random.sample(wrong_options, 4)

        all_options = selected_wrong + [correct_class]
        random.shuffle(all_options)  # Shuffle to randomize option order

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
    csv_file = 'OCT5k.csv'
    result = generate_questions(csv_file)

    with open('OCT5k\OCT5k.json', 'w') as f:
        json.dump(result, f, indent=4)
    print("Generated questions saved to 'OCT5k.json'")
