import os
import pandas as pd
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Define the paths for the image folder and json folder
image_folder = r'YOUR_PATH_HERE\G1020\Images'  # Replace with the actual path to the image folder
json_folder = r'YOUR_PATH_HERE\G1020\json'  # Replace with the actual path to the json folder
output_folder = r'YOUR_PATH_HERE\G1020\cup'  # Define the output folder to save the processed images

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to process and save images with 'cup' label
def plot_cup_labels(image_folder, json_folder, output_folder):
    # Initialize counter for naming files
    counter = 1

    # Traverse through all image files in the folder
    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):  # Process only image files
            # Construct corresponding JSON file name
            json_name = os.path.splitext(image_name)[0] + '.json'
            json_path = os.path.join(json_folder, json_name)
            image_path = os.path.join(image_folder, image_name)

            if os.path.exists(json_path):
                # Load the image
                img = Image.open(image_path)

                # Load the corresponding JSON data
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Check if "disc" label exists in the JSON data
                for shape in data['shapes']:
                    if shape['label'] == 'cup':
                        points = shape['points']

                        # Create a blank canvas with the original image
                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.imshow(img)
                        # Draw a polygon with only white border, no fill
                        polygon = patches.Polygon(points, closed=True, edgecolor='white', linewidth=4, fill=False)
                        ax.add_patch(polygon)

                        # Hide axes and set tight layout
                        ax.axis('off')
                        plt.tight_layout()

                        # Save the figure
                        output_image_path = os.path.join(output_folder, f'cup_{counter}.png')
                        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.0, transparent=False)
                        plt.close(fig)

                        # Resize the image to 512x512
                        img_resized = Image.open(output_image_path)
                        img_resized = img_resized.resize((512, 512), Image.Resampling.LANCZOS)
                        img_resized.save(output_image_path)

                        # Increment counter
                        counter += 1
                        break  # Exit after processing 'disc' label

# Run the function to process images and json files, and save results to the output folder
plot_cup_labels(image_folder, json_folder, output_folder)

# Define the image folder path
# image_folder = r'YOUR_PATH_HERE\Fun_Fundus_OCT\location\G1020\disc'  # Replace with the actual path to the image folder
# #
# # Initialize an empty list to store image file paths
# image_data = []
#
# # Traverse through all image files in the folder
# for image_name in os.listdir(image_folder):
#     if image_name.endswith('.jpg') or image_name.endswith('.png'):  # Process only image files
#         # Construct the file path
#         image_path = f'images/{image_name}'  # Add 'images/' before the image name
#         image_data.append({'new_id': image_path})
#
# # Convert the list of image paths into a DataFrame
# df = pd.DataFrame(image_data)
#
# # Define the output CSV file path
# output_csv_path = r'YOUR_PATH_HERE\Fun_Fundus_OCT\location\G1020\G1020.csv'  # Replace with your desired output path
#
# # Save the DataFrame to a CSV file
# df.to_csv(output_csv_path, index=False)
#
# print(f"CSV file has been saved to {output_csv_path}")
#
# # 定义所有可能的选项
# full_options = ["arteries", "veins", "microvascular", "optic disk", "optic cup"]
#
# def generate_questions(csv_file):
#     # 加载 CSV 文件
#     df = pd.read_csv(csv_file)
#
#     result = []
#     for index, row in df.iterrows():
#         new_id = row['new_id']
#         correct_answer = row['class']
#
#         # 确保正确答案在选项列表中
#         if correct_answer not in full_options:
#             print(f"Invalid class '{correct_answer}' for {new_id}, skipping...")
#             continue
#
#         # 生成错误选项：从 full_options 中排除正确答案后随机选择 4 个
#         wrong_options = [opt for opt in full_options if opt != correct_answer]
#         selected_wrong = random.sample(wrong_options, 4)
#
#         # 组合所有选项：4 个错误选项 + 1 个正确答案
#         all_options = selected_wrong + [correct_answer]
#         random.shuffle(all_options)  # 随机打乱选项顺序
#
#         # 生成问题字符串
#         letters = list("ABCDE")
#         question = "Identify which category the white-labeled structures in the fundus image belong to?\n"
#         for letter, opt in zip(letters, all_options):
#             question += f"{letter}. {opt}\n"
#
#         # 确定正确答案的字母
#         idx = all_options.index(correct_answer)
#         correct_letter = letters[idx]
#
#         # 构造问题字典
#         q_dict = {
#             "id": new_id,  # 直接使用 new_id（如 "images/arteries_12.png"）
#             "question": question.strip(),
#             "answer": correct_letter
#         }
#
#         result.append(q_dict)
#
#     return result
#
# if __name__ == "__main__":
#     # 路径设置
#     csv_file = r'YOUR_PATH_HERE\Fun_Fundus_OCT\location\G1020\G1020.csv'
#     output_json = r'YOUR_PATH_HERE\Fun_Fundus_OCT\location\G1020\G1020.json'
#
#     # 确保输出目录存在
#     os.makedirs(os.path.dirname(output_json), exist_ok=True)
#
#     # 生成问题
#     result = generate_questions(csv_file)
#
#     # 保存到 JSON 文件
#     with open(output_json, 'w') as f:
#         json.dump(result, f, indent=4)
#     print(f"Generated questions saved to '{output_json}'")
