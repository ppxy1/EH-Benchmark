import os
import json
import base64
from openai import OpenAI
from tqdm import tqdm
import csv
import time
import json
import cv2
import numpy as np
import pandas as pd
import os

import os
import pandas as pd
import numpy as np
from PIL import Image

# # 设置路径
# csv_path = r'FGADR.csv'
# mask_folder = r'HardExudate_Masks'
# output_csv_path = r'FGADR.csv'
#
# # 加载CSV
# df = pd.read_csv(csv_path)
#
# # 遍历掩码文件夹中的每张图像
# for filename in os.listdir(mask_folder):
#     if not filename.lower().endswith(".png"):
#         continue
#
#     base_name = os.path.splitext(filename)[0]
#     mask_path = os.path.join(mask_folder, filename)
#
#     # 在 CSV 中找到对应行
#     matched_row = df[df['new_id'].str.contains(base_name)]
#     if matched_row.empty:
#         continue  # 没有对应记录
#
#     index = matched_row.index[0]
#     position = matched_row.iloc[0]['position']
#
#     try:
#         mask = Image.open(mask_path).convert("L")
#         mask_np = np.array(mask)
#     except Exception:
#         df.at[index, 'IRMA'] = 'Mask not loaded'
#         continue
#
#     # 创建二值掩码
#     mask_binary = mask_np == 255
#
#     # 图像中心
#     h, w = mask_binary.shape
#     cx = w // 2
#     cy = h // 2
#
#     # 区域掩码
#     superior_mask = np.zeros_like(mask_binary, dtype=bool)
#     superior_mask[:cy, :] = True
#     inferior_mask = np.zeros_like(mask_binary, dtype=bool)
#     inferior_mask[cy:, :] = True
#
#     if position == 'OD':
#         nasal_mask = np.zeros_like(mask_binary, dtype=bool)
#         nasal_mask[:, cx:] = True
#         temporal_mask = np.zeros_like(mask_binary, dtype=bool)
#         temporal_mask[:, :cx] = True
#     else:  # OS
#         nasal_mask = np.zeros_like(mask_binary, dtype=bool)
#         nasal_mask[:, :cx] = True
#         temporal_mask = np.zeros_like(mask_binary, dtype=bool)
#         temporal_mask[:, cx:] = True
#
#     # 判断病灶位置
#     irma_regions = []
#     if np.any(mask_binary & superior_mask):
#         irma_regions.append('Superior')
#     if np.any(mask_binary & inferior_mask):
#         irma_regions.append('Inferior')
#     if np.any(mask_binary & nasal_mask):
#         irma_regions.append('Nasal')
#     if np.any(mask_binary & temporal_mask):
#         irma_regions.append('Temporal')
#
#     if not irma_regions:
#         df.at[index, 'HardExudate'] = 'None'
#     else:
#         df.at[index, 'HardExudate'] = ', '.join(sorted(irma_regions))
#
# # 保存更新后的 CSV
# df.to_csv(output_csv_path, index=False)
# print(f"处理完成，结果已保存到：'{output_csv_path}'")

df = pd.read_csv(r'FGADR.csv')
print(df['HardExudate'].unique())

import pandas as pd
import random
import json

# 定义所有可能的选项
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
'None']

def generate_questions(csv_file):
    # 加载CSV文件
    df = pd.read_csv(csv_file)

    # 筛选IRMA列有数据的行（非空且不包含错误信息）
    df = df[df['HardExudate'].notna() & ~df['HardExudate'].isin(['Mask not found', 'Mask not loaded'])]

    result = []
    for index, row in df.iterrows():
        new_id = row['new_id']
        irma_location = row['HardExudate']

        # 如果IRMA列为“None”，跳过
        if irma_location == 'None':
            continue

        # 确保IRMA值在选项列表中
        if irma_location not in full_options:
            continue

        # 生成错误选项：从full_options中排除正确答案后随机选择4个
        wrong_options = [opt for opt in full_options if opt != irma_location]
        selected_wrong = random.sample(wrong_options, 4)

        # 组合所有选项：4个错误选项 + 1个正确答案
        all_options = selected_wrong + [irma_location]
        random.shuffle(all_options)  # 随机打乱选项顺序

        # 生成问题字符串
        letters = list("ABCDE")
        question = "Where does Hard exudate occur?\n"
        for letter, opt in zip(letters, all_options):
            question += f"{letter}. {opt}\n"

        # 确定正确答案的字母
        idx = all_options.index(irma_location)
        correct_letter = letters[idx]

        # 构造问题字典
        q_dict = {
            "id": f"images/{new_id}",
            "question": question.strip(),
            "answer": correct_letter
        }

        result.append(q_dict)

    return result


if __name__ == "__main__":
    csv_file = r'FGADR.csv'
    result = generate_questions(csv_file)

    with open(r'HardExudate.json', 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Generated questions saved to 'HardExudate.json'")