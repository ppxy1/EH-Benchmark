import pandas as pd
import numpy as np
import json
import os
import shutil

file_path = "path/to/VF_and_clinical_information.xlsx"
baseline_df = pd.read_excel(file_path, sheet_name="Baseline")
baseline_df.columns = baseline_df.columns.str.strip()

vf_columns = baseline_df.columns[-61:]

RNFL_MEAN_COL = "Mean"
RNFL_S_COL = "S"
RNFL_N_COL = "N"
RNFL_I_COL = "I"
RNFL_T_COL = "T"

required_columns = ["Subject Number", "Laterality", "Age", "Gender", "IOP", "CCT",
                    "Category of Glaucoma", RNFL_MEAN_COL, RNFL_S_COL, RNFL_N_COL,
                    RNFL_I_COL, RNFL_T_COL, "Corresponding CFP"]

missing_columns = [col for col in required_columns if col not in baseline_df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in dataset: {missing_columns}")

def calculate_vf_mean(row, vf_columns):
    try:
        return row[vf_columns].mean()
    except:
        return np.nan

def generate_question_answer(subject_num, laterality):
    baseline_row = baseline_df[(baseline_df["Subject Number"] == subject_num) &
                               (baseline_df["Laterality"] == laterality)].iloc[0]
    try:
        age = baseline_row["Age"]
        gender = baseline_row["Gender"]
        iop = baseline_row["IOP"]
        cct = baseline_row["CCT"]
        rnfl_mean = baseline_row[RNFL_MEAN_COL]
        rnfl_s = baseline_row[RNFL_S_COL]
        rnfl_n = baseline_row[RNFL_N_COL]
        rnfl_i = baseline_row[RNFL_I_COL]
        rnfl_t = baseline_row[RNFL_T_COL]
        cfp = baseline_row["Corresponding CFP"]
        glaucoma_category = baseline_row["Category of Glaucoma"]
    except KeyError as e:
        raise

    vf_mean = calculate_vf_mean(baseline_row, vf_columns)
    vf_mean_str = "unavailable" if np.isnan(vf_mean) else f"{vf_mean:.1f}"

    def format_value(value):
        try:
            return f"{float(value):.1f}"
        except:
            return "unavailable"

    rnfl_mean_str = format_value(rnfl_mean)
    rnfl_s_str = format_value(rnfl_s)
    rnfl_n_str = format_value(rnfl_n)
    rnfl_i_str = format_value(rnfl_i)
    rnfl_t_str = format_value(rnfl_t)

    question = f"""
For subject {subject_num}, {laterality}, a {age}-year-old {gender}, baseline clinical indicators include
IOP of {iop:.1f} mmHg, CCT of {cct} μm, OCT RNFL thickness (Mean: {rnfl_mean_str} μm;
Superior: {rnfl_s_str} μm, Nasal: {rnfl_n_str} μm, Inferior: {rnfl_i_str} μm, Temporal: {rnfl_t_str} μm),
and mean visual field sensitivity of {vf_mean_str} dB. Based on these indicators and the optic disc ROI
in the fundus image, what is the most likely Category of Glaucoma?

A. Open-Angle Glaucoma (OAG)
B. Angle-Closure Glaucoma (ACG)
C. Normal Tension Glaucoma (NTG)
D. Secondary Glaucoma
E. No Glaucoma
"""

    if glaucoma_category == "OAG":
        answer = "A"
    elif glaucoma_category == "ACG":
        answer = "B"
    elif glaucoma_category == "NTG":
        answer = "C"
    elif glaucoma_category == "Secondary":
        answer = "D"
    else:
        answer = "E"

    return question, answer, cfp

unique_subjects = baseline_df[["Subject Number", "Laterality"]].drop_duplicates()

json_data = []
for _, row in unique_subjects.iterrows():
    subject_num = row["Subject Number"]
    laterality = row["Laterality"]
    question, answer, cfp = generate_question_answer(subject_num, laterality)
    json_data.append({
        "id": f"images/{cfp}",
        "question": question,
        "answer": answer
    })

with open("path/to/GRAPE.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4)

print(f"Total samples generated: {len(json_data)}")

excel_file = "path/to/VF_and_clinical_information.xlsx"
source_folder = "path/to/Annotated"
target_folder = "path/to/images"

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

baseline_df = pd.read_excel(excel_file, sheet_name="Baseline")
baseline_df.columns = baseline_df.columns.str.strip()
image_files = [str(f).strip() for f in baseline_df["CFP"].dropna().unique()]
actual_files = set(os.listdir(source_folder))

copied_count = 0
for image_file in image_files:
    source_path = os.path.join(source_folder, image_file)
    target_path = os.path.join(target_folder, image_file)
    if image_file in actual_files:
        shutil.copy2(source_path, target_path)
        copied_count += 1
    else:
        print(f"Warning: {image_file} not found in {source_folder}")

print(f"Total unique images in Excel: {len(image_files)}")
print(f"Successfully copied: {copied_count} images")
print(f"Failed to copy: {len(image_files) - copied_count} images")
