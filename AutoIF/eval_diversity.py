import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import argparse
import json
from datasets import load_dataset
from tqdm import tqdm


# Utility functions
def load_dataset_from_file(filename):
    """
    Load a dataset from a file. Supports both .json and .jsonl formats.
    """
    if filename.endswith('.json'):
        with open(filename, 'r') as file:
            return json.load(file)
    elif filename.endswith('.jsonl'):
        return load_jsonl_to_list(filename)
    else:
        raise ValueError("Invalid file format. Please provide a .json or .jsonl file.")


def load_jsonl_to_list(jsonl_file_path):
    """
    Helper function to load data from a .jsonl file into a list of dictionaries.
    """
    data_list = []
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            data_list.append(json_obj)
    return data_list


################
# Configuration
################
def get_args():
    """
    Argument parsing for the script. 
    """
    parser = argparse.ArgumentParser(description="Similarity Calculation Manager.")
    
    # Experiment Settings
    parser.add_argument("--sentence_model", type=str, default="<path_to_model>/mpnet-base", 
                        help="Path to the sentence model")
    parser.add_argument("--input_file", type=str, default="<path_to_data>/autoif-small.json", 
                        help="Input dataset file name")
    parser.add_argument("--encoding_batch_size", type=int, default=65536, 
                        help="Batch size for encoding the sentences.")
    parser.add_argument("--distance_threshold", type=float, default=0.05, 
                        help="Distance threshold for the similarity search.")
    parser.add_argument("--search_space_size", type=int, default=500, 
                        help="Number of examples to search for similarity.")
    parser.add_argument("--search_batch_size", type=int, default=1024, 
                        help="Batch size for searching for similarity.")

    # System Settings
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--save_faiss_index", type=bool, default=True, 
                        help="Save the Faiss index.")
    
    return parser.parse_args()


args = get_args()

# Model and Dataset Paths
sentence_model = args.sentence_model
dataset_path = args.input_file
dataset_name = dataset_path.split("/")[-1].split(".")[0]  # Extract dataset name from path
output_file = f"results/{dataset_name}_distance.jsonl"  # Output file path

################
# Step 1 - Load the Dataset and Build the Faiss Index
################
# Load the dataset
print(f"Loading dataset from {dataset_path}...")
dataset = load_dataset("json", data_files=dataset_path)

# Prepare input sentences (instruction + input)
inputs = [dataset["train"]["instruction"][i] + "\n" + dataset["train"]["input"][i] 
          for i in range(len(dataset["train"]))]
print(f"The second instruction in the dataset is: {inputs[1]}")

# Load the sentence transformer model
model = SentenceTransformer(sentence_model)
model.to(device=f'cuda:{args.device}', dtype=torch.float32)
print(f"The model is loaded on device: {model.device}")

# Encode the sentences into vectors in batches
encoding_batch_size = args.encoding_batch_size
embeddings = []
for i in tqdm(range(0, len(inputs), encoding_batch_size), desc="Encoding Sentences"):
    batch_sentences = inputs[i:i+encoding_batch_size]
    batch_embeddings = model.encode(batch_sentences, convert_to_tensor=True, show_progress_bar=True)
    embeddings.append(batch_embeddings.cpu().numpy())

# Concatenate the embeddings into a single numpy array
embeddings = np.concatenate(embeddings, axis=0)
print(f"The shape of the concatenated embeddings is: {embeddings.shape}")

# Add the embeddings to the dataset
print("Adding embeddings to the dataset...")
dataset["train"] = dataset["train"].add_column("embeddings", embeddings.tolist())

# Build the Faiss index
print("Building the Faiss index...")
dataset["train"].add_faiss_index(column="embeddings")

# Save the Faiss index to disk
if args.save_faiss_index:
    print("Saving the Faiss index...")
    index = dataset["train"].get_index("embeddings")
    faiss_index = index.faiss_index
    index_file = f"results/{dataset_name}.faiss"
    faiss.write_index(faiss_index, index_file)

################
# Step 2 - Find Similar Examples
################
distance_threshold = args.distance_threshold
search_space_size = args.search_space_size
search_batch_size = args.search_batch_size
n_batches = (len(dataset["train"]) + search_batch_size - 1) // search_batch_size
print(f"Number of batches: {n_batches}")

# Load the original dataset from file for output
unfilled_dataset = load_dataset_from_file(dataset_path)

# Perform similarity search for each batch and write results
with open(output_file, 'a', encoding="utf-8") as file:
    for batch_idx in tqdm(range(n_batches), desc="Processing Batches"):
        start_idx = batch_idx * search_batch_size
        end_idx = min((batch_idx + 1) * search_batch_size, len(dataset["train"]))

        batch_indices = list(range(start_idx, end_idx))
        batch_embeddings = embeddings[batch_indices]
        
        # Perform the similarity search
        search_results = dataset["train"].search_batch(queries=batch_embeddings, k=search_space_size, index_name="embeddings")
        total_scores = search_results.total_scores
        total_indices = search_results.total_indices

        # Process the results for each query
        for i in range(len(total_indices)):
            scores = total_scores[i]
            indices = total_indices[i]
            min_distance = float(scores[1])  # Exclude the query itself (index 0)

            # Add min_distance to the dataset
            dataset["train"][start_idx + i]["min_distance"] = min_distance

            # Filter based on the distance threshold and exclude the query itself
            filtered_indices = [index for index, score in zip(indices, scores) if score < distance_threshold]
            filtered_indices = [index for index in filtered_indices if index != start_idx + i]

            repeat_count = len(filtered_indices) if filtered_indices else 0
            dataset["train"][start_idx + i]["repeat_count"] = repeat_count

            # Write the updated example to the output file
            line = unfilled_dataset[start_idx + i]
            line["min_neighbor_distance"] = min_distance
            line["repeat_count"] = repeat_count
            file.write(json.dumps(line) + '\n')
        
        print(f"Batch {batch_idx} results saved to output file.")

print("Distance calculation completed.")
