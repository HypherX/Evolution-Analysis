import jsonlines
import json
import random
import re
import os
import logging
import signal
import numpy as np
from concurrent.futures import TimeoutError
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    RetryError
)

# Set random seed for reproducibility
random.seed(0)

# Configuration: Set the input and output file paths
INPUT_FILE_PATH = "data/eval_func_rft_large.jsonl"  # Path to the input JSONL file
OUTPUT_FILE_PATH = "data/cross_validation_large.jsonl"  # Path to save the output JSONL file

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Test GPT-4 (Placeholder for your model testing)
os.environ['NLTK_DATA'] = '/root/nltk_data'  # Set NLTK data path for the environment
logging.getLogger('nltk').setLevel(logging.CRITICAL)
import nltk
from nltk import data
data.path.append('/root/nltk_data')


def preprocess_verification_functions(results):
    """Preprocess verification functions, collecting imports and checking validity."""
    collect_packages = []
    for result in results:
        res = result['gpt-answer']
        eval_funcs = []
        for each in res:
            try:
                # Extract the JSON block
                json_dict = re.findall(r'```json(.*?)```', each, re.DOTALL)[0].strip()
            except IndexError:
                continue
            
            # Try to parse the JSON and validate the function
            try:
                res_dict = json.loads(json_dict)
            except json.JSONDecodeError:
                continue
            
            func = res_dict['func']
            
            if '\\n' in func:
                func = func.replace('\\n', '\n')
            
            try:
                # Execute the function to check for validity
                exec(func)
            except Exception:
                continue
            
            # Collect packages related to the function
            for line in func.split('\n'):
                if 'import' in line or 'download' in line or 'requests' in line:
                    collect_packages.append(line)
    return collect_packages


def timeout_handler(signum, frame):
    """Handle timeout for function execution."""
    raise TimeoutError("Function execution timed out")


def cross_validate_functions(results):
    """Cross-validate the generated functions using the test cases."""
    filter_results = []
    for result in tqdm(results):
        res = result['gpt-answer']
        eval_funcs = []
        test_cases = []
        for each in tqdm(res):
            try:
                # Extract the JSON block
                json_dict = re.findall(r'```json(.*?)```', each, re.DOTALL)[0].strip()
            except IndexError:
                continue

            try:
                res_dict = json.loads(json_dict)
            except json.JSONDecodeError:
                continue

            # Process and clean the function
            func = res_dict['func']
            func = func.strip()
            func = '\n'.join([line for line in func.split('\n') if 'download' not in line and 'requests' not in line])
            
            try:
                exec(func)
            except Exception:
                continue

            eval_funcs.append(func)

            # Extract test cases from the result
            try:
                for case in res_dict['cases']:
                    try:
                        test_cases.append((case['input'], case['output']))
                    except KeyError:
                        logging.warning("Test case missing 'input' or 'output': %s", case)
            except KeyError:
                logging.warning("No 'cases' key found in result: %s", res_dict)

        # Deduplicate functions and test cases
        eval_funcs = list(set(eval_funcs))
        test_cases = list(map(json.loads, set(map(json.dumps, test_cases))))

        # Skip if no valid functions or test cases
        if not eval_funcs or not test_cases:
            continue

        filtered_test_cases = []
        for case in tqdm(test_cases):
            flag = False
            for func in eval_funcs:
                local_vars = {}
                try:
                    exec(func, globals(), local_vars)
                except Exception:
                    continue

                if 'evaluate' not in local_vars:
                    continue

                eval_func = local_vars['evaluate']
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(5)  # Timeout after 5 seconds
                    res = eval_func(case[0])
                except Exception:
                    res = None
                finally:
                    signal.alarm(0)

                if res is not None and res == case[1]:
                    flag = True
            if flag:
                filtered_test_cases.append(case)

        # Score each function based on its accuracy
        scored_funcs = []
        for func in tqdm(eval_funcs):
            local_vars = {}
            try:
                exec(func, globals(), local_vars)
            except Exception:
                continue

            if 'evaluate' not in local_vars:
                continue

            eval_func = local_vars['evaluate']
            acc = []
            for inp, out in filtered_test_cases:
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(5)
                    res = eval_func(inp)
                except Exception:
                    res = None
                finally:
                    signal.alarm(0)

                acc.append(1 if res == out else 0)

            accuracy = np.mean(acc) if acc else 0
            scored_funcs.append((func, accuracy))

        # Filter out functions with low accuracy
        valid_funcs = [func for func, acc in scored_funcs if acc >= 0.8]
        if not valid_funcs:
            continue

        filter_results.append({
            "instruction": result['instruction'],
            "eval_func": valid_funcs,
            "cases": filtered_test_cases
        })

    return filter_results


def main():
    """Main function to load data, process functions, and perform cross-validation."""
    # Load the results from the input file
    results = list(jsonlines.open(INPUT_FILE_PATH))

    # Step 1: Preprocess the verification functions and collect packages
    logging.info("Preprocessing verification functions...")
    collect_packages = preprocess_verification_functions(results)
    logging.info("Collected packages: %s", list(set(collect_packages)))

    # Step 2: Perform cross-validation for the functions and cases
    logging.info("Performing cross-validation...")
    filter_results = cross_validate_functions(results)

    # Step 3: Save the filtered results to the output file
    logging.info("Saving filtered results to %s...", OUTPUT_FILE_PATH)
    with jsonlines.open(OUTPUT_FILE_PATH, "w") as f:
        for item in filter_results:
            f.write(item)

    logging.info("Finished processing!")


if __name__ == "__main__":
    main()
