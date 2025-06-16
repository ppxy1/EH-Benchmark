import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import random
import shutil
import csv

def convert_to_seven_class_label(image_id, base_dir):
    label_image = np.zeros((1024, 1024), dtype=np.uint8)
    categories = [
        "IRMA_Masks",
        "Hemohedge_Masks",
        "Microaneurysms_Masks",
        "Neovascularization_Masks",
        "SoftExudate_Masks",
        "HardExudate_Masks"
    ]
    for idx, category in enumerate(categories, start=1):
        category_dir = os.path.join(base_dir, category)
        mask_path = os.path.join(category_dir, f"{image_id}.png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((1024, 1024))
            mask_array = np.array(mask)
            label_image[mask_array > 0] = idx
    return label_image

def save_seven_class_label(image_id, label_image, output_dir):
    output_path = os.path.join(output_dir, f"{image_id}.png")
    Image.fromarray(label_image).save(output_path)

def process_all_images(base_dir, output_dir):
    image_ids = set()
    categories = [
        "IRMA_Masks",
        "Hemohedge_Masks",
        "Microaneurysms_Masks",
        "Neovascularization_Masks",
        "SoftExudate_Masks",
        "HardExudate_Masks"
    ]
    for category in tqdm(categories):
        category_dir = os.path.join(base_dir, category)
        for filename in os.listdir(category_dir):
            if filename.endswith(".png"):
                image_id = filename.split('.')[0]
                image_ids.add(image_id)
    for image_id in image_ids:
        label_image = convert_to_seven_class_label(image_id, base_dir)
        save_seven_class_label(image_id, label_image, output_dir)

def resize_image(image_path, output_path, size=(512, 512)):
    img = Image.open(image_path)
    img_resized = img.resize(size)
    img_resized.save(output_path)

def resize_all_images(input_dir, output_dir, size=(512, 512)):
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            resize_image(image_path, output_path, size)

def create_train_val_csv(images_dir, masks_dir, output_train_csv, output_val_csv, train_ratio=0.9):
    images = sorted(os.listdir(images_dir))
    masks = sorted(os.listdir(masks_dir))
    assert len(images) == len(masks)
    data = []
    for image, mask in zip(images, masks):
        relative_image_path = os.path.join("FGADR", "images", image)
        relative_mask_path = os.path.join("FGADR", "masks", mask)
        data.append([relative_image_path, relative_mask_path])
    random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    train_df = pd.DataFrame(data[:train_size], columns=["Image Path", "Mask Path"])
    val_df = pd.DataFrame(data[train_size:], columns=["Image Path", "Mask Path"])
    train_df.to_csv(output_train_csv, index=False)
    val_df.to_csv(output_val_csv, index=False)

def extract_images_to_flat_dir(base_dir, image_dir, csv_file, categories):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'category'])
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(base_dir, split)
            for category in categories:
                category_dir = os.path.join(split_dir, category)
                if os.path.exists(category_dir):
                    for img_name in os.listdir(category_dir):
                        img_path = os.path.join(category_dir, img_name)
                        if os.path.isfile(img_path):
                            img_dest = os.path.join(image_dir, img_name)
                            shutil.copy(img_path, img_dest)
                            writer.writerow([img_name, category])

def prefix_image_path_in_csv(csv_path, output_csv_path, prefix='images/'):
    df = pd.read_csv(csv_path)
    if 'image_name' in df.columns:
        df['image_name'] = prefix + df['image_name'].astype(str)
        df.to_csv(output_csv_path, index=False)
