import pandas as pd
import os
from PIL import Image
import numpy as np
import random
import json

# # 读取CSV文件
# df = pd.read_csv(r'IDRID.csv')
# mask_folder = r'4. Soft Exudates'
# mask_files = os.listdir(mask_folder)
#
# for mask_file in mask_files:
#     # 提取掩码文件的基本名称（去掉扩展名和"_MA"后缀）
#     base_name = os.path.splitext(mask_file)[0]  # e.g., "IDRiD_01_MA"
#     core_name = base_name.replace('_SE', '')  # e.g., "IDRiD_01"
#
#     # 在CSV中查找匹配的行
#     # 从new_id中提取核心名称（去掉路径和扩展名）
#     matching_row = df[df['new_id'].apply(lambda x: os.path.splitext(os.path.basename(x))[0]) == core_name]
#     if matching_row.empty:
#         print(f"未找到匹配的行: {core_name}")
#         continue
#
#     index = matching_row.index[0]
#     position = matching_row['position'].values[0]
#
#     # 读取掩码图像
#     mask_path = os.path.join(mask_folder, mask_file)
#     mask_image = Image.open(mask_path)
#
#     # 如果是RGB图像，提取红色通道并转换为二值掩码
#     if mask_image.mode == 'RGB':
#         mask_array = np.array(mask_image)
#         # 提取红色通道（假设微血管瘤是红色点）
#         red_channel = mask_array[:, :, 0]  # 红色通道
#         mask_binary = (red_channel > 0)  # 红色点视为病灶
#     else:
#         # 如果是灰度图像，直接转换
#         mask_image = mask_image.convert('L')
#         mask_array = np.array(mask_image)
#         mask_binary = (mask_array == 255)
#
#     # 计算图像中心
#     height, width = mask_binary.shape
#     cy = height // 2
#     cx = width // 2
#
#     # 生成区域掩码
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
#     # 更新DataFrame
#     if not irma_regions:
#         df.at[index, 'Soft Exudate'] = 'None'
#     else:
#         df.at[index, 'Soft Exudate'] = ', '.join(sorted(irma_regions))
#
# # 保存更新后的CSV
# df.to_csv(r'IDRID.csv', index=False)
# print("处理完成，结果已保存到 'IDRID_updated.csv'")

df = pd.read_csv(r'IDRID.csv')
print(df['Soft Exudate'].unique())

# 定义所有可能的选项
full_options = [
'Inferior, Nasal, Superior, Temporal',
'Inferior, Superior, Temporal',
'Inferior, Nasal, Superior',
'Nasal, Superior, Temporal',
'Superior, Temporal',
'Nasal, Superior',
'Inferior, Temporal',
'Inferior, Nasal, Temporal',
'None']

def generate_questions(csv_file):
    # 加载CSV文件
    df = pd.read_csv(csv_file)

    # 筛选IRMA列有数据的行（非空且不包含错误信息）
    df = df[df['Soft Exudate'].notna() & ~df['Soft Exudate'].isin(['Mask not found', 'Mask not loaded'])]

    result = []
    for index, row in df.iterrows():
        new_id = row['new_id']
        irma_location = row['Soft Exudate']

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
        question = "Where does Soft Exudate occur?\n"
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
    csv_file = r'IDRID.csv'
    result = generate_questions(csv_file)

    with open(r'Soft Exudate.json', 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Generated questions saved to 'Soft Exudate.json'")
