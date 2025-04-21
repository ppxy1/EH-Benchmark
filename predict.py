import os
import json
import csv
import re
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = "https://api.deepseek.com"

# 检查是否提供了 API 密钥
if not api_key:
    raise ValueError("Missing DeepSeek API key. Please set DEEPSEEK_API_KEY in your environment.")

# 初始化 DeepSeek 客户端
client = OpenAI(api_key=api_key, base_url=base_url)

# 加载 JSON 数据
with open('ODIR5k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 写入 CSV
with open('ODIR5k_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['id', 'question', 'correct_answer', 'model_prediction']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    correct_count = 0
    total = len(data)

    for entry in tqdm(data, desc="Evaluating"):
        id = entry['id']
        question = entry['question']
        correct_answer = entry['answer']

        prompt = f"{question} Respond with only the letter of the correct answer (A, B, C, D, or E)."
        messages = [
            {"role": "system", "content": "You are a helpful assistant who answers multiple-choice questions by choosing A, B, C, D, or E."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = client.chat.completions.create(
                model='deepseek-chat',
                messages=messages,
                stream=False
            )
            model_response = response.choices[0].message.content.strip()

            match = re.search(r'\b[A-E]\b', model_response, re.IGNORECASE)
            model_prediction = match.group(0).upper() if match else "ERROR"

        except Exception as e:
            print(f"Error for id {id}: {e}")
            model_prediction = "ERROR"

        writer.writerow({
            'id': id,
            'question': question,
            'correct_answer': correct_answer,
            'model_prediction': model_prediction
        })

        if model_prediction == correct_answer:
            correct_count += 1

# 输出准确率
accuracy = correct_count / total if total > 0 else 0
print(f"Accuracy: {accuracy:.2%}")