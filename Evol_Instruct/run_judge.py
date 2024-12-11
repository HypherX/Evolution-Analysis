import json
import concurrent.futures
import time
import tqdm
import argparse
from tqdm import tqdm
from multiprocessing import Process as mp
from multiprocessing import Pool
import pyarrow.parquet as pq
import pandas as pd
import openai
from openai import AzureOpenAI

import tqdm
import re

openai.api_type = "azure"
openai.api_base = "https://baaisolution-ae.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "2d196d16e1bc470c86689efdc7ca4943"
deployment_name='llmsft'

def call_chatgpt_azure_format(query):
    '''print("=================")
    print(query)
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    )'''
    responses = []
    endpoint = "https://baaisolution-ae.openai.azure.com/"
    client = AzureOpenAI(
        api_version="2023-07-01-preview",
        azure_endpoint=endpoint,
        api_key="2d196d16e1bc470c86689efdc7ca4943",
    )
    response = client.chat.completions.create(
        model="gpt-35-turbo",
        stream=False,
        messages=[
            {"role": "user", "content": query}
    ])

    responses.append(response.choices[0].message.content)
    return responses
# Define command-line arguments
parser = argparse.ArgumentParser(description="Compare AI assistant responses.")
parser.add_argument('--predict_data_file', type=str, default="data/llama3-icifd-15%.json", help="Path to the predict data file 1")
parser.add_argument('--predict_data_file_another', type=str, default="data/llama3-ifd-15%.json", help="Path to the predict data file 2")
args = parser.parse_args()

# Load prediction data

with open(args.predict_data_file, "r") as r:
    predict_data = json.load(r)

# Load reference answers
with open(args.predict_data_file_another, "r") as r:
    predict_data_another = json.load(r)

for i in range(len(predict_data)):
    predict_data[i]["another"] = predict_data_another[i]["output"]
 
PROMPT_eval_single="""
[User]
{user_query}
[End of User]
[Assistant 1]
{assistant1}
[End of Assistant 1]
[Assistant 2]
{assistant2}
[End of Assistant 2]
[System]
We would like to request your feedback on the two dialogues shown above between an AI assistant and a user.
Focus on the AI's responses. The AI's responses should perfectly align with the user's needs. Additionally, the responses should be concise and to the point, avoiding unnecessary details or excessive information, while still being as comprehensive as possible in addressing the user's query. The answers must maintain good logical flow, use precise technical terms, and be factually accurate and objective.
Based on the above criteria, compare the performance of Assistant 1 and Assistant 2. Determine which one is "better than," "worse than," or "equal to" the other. First, compare their responses and analyze which aligns better with the stated requirements.

On the last line, output a single label only, selecting from one of the following:
'Assistant 1 is better than Assistant 2'
'Assistant 1 is worse than Assistant 2'
'Assistant 1 is equal to Assistant 2'
"""

# Initialize counters
win_number = 0
loss_number = 0
tie_number = 0
total_number_single = 0
total_number_multi = 0
total_number_choice = 0

def call_with_retry(user_input, retries=50, delay=5):
    for _ in range(retries):
        try:
            return call_chatgpt_azure_format(user_input)[0]
        except:
            time.sleep(delay)
    raise Exception("Max retries exceeded")

def process_item(item):
    global win_number, loss_number, tie_number, total_number_single, total_number_multi, total_number_choice
    total_number_single += 1
    if item["input"]:
        user_input1 = PROMPT_eval_single.format(user_query=item["instruction"] + "\n" + item["input"], assistant1=item["output"], assistant2=item["another"])
        user_input2 = PROMPT_eval_single.format(user_query=item["instruction"] + "\n" + item["input"], assistant2=item["output"], assistant1=item["another"])
    else:
        user_input1 = PROMPT_eval_single.format(user_query=item["instruction"], assistant1=item["output"], assistant2=item["another"])
        user_input2 = PROMPT_eval_single.format(user_query=item["instruction"], assistant2=item["output"], assistant1=item["another"])
    try:
        resp1 = call_with_retry(user_input1)
        resp2 = call_with_retry(user_input2)
    except Exception as e:
        print(f"Skipping item due to error: {e}")
        return

    if "Assistant 1 is better than Assistant 2" in resp1 and "Assistant 1 is worse than Assistant 2" in resp2:
        win_number += 1
    elif "Assistant 1 is worse than Assistant 2" in resp1 and "Assistant 1 is better than Assistant 2" in resp2:
        loss_number += 1
    else:
        tie_number += 1
        
# Process items with multiple threads
num_threads = 1

with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    list(tqdm.tqdm(executor.map(process_item, predict_data), total=len(predict_data)))


print(total_number_single, total_number_multi, total_number_choice)

# Calculate total number of judgments
total_num = win_number + tie_number + loss_number

# Print results
print("Winning Rate: ", win_number / total_num, "Losing Rate: ", loss_number / total_num, "Tie Rate: ", tie_number / total_num)
