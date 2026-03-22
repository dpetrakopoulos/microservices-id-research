import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
JSON_DATA_FILE = 'petclinic_data.json'
ASSIGNMENTS_FILE = 'microservice_assignments.csv'
MODEL_NAME = "microsoft/codebert-base"


def get_codebert_embedding(text, tokenizer, model):
    """Generates a dense vector embedding for a piece of code using CodeBERT."""
    # Tokenize the text (truncate to 512 tokens as required by BERT architectures)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # Pass through the model without calculating gradients (saves memory/time)
    with torch.no_grad():
        outputs = model(**inputs)

    # We use the [CLS] token representation (index 0) as the embedding for the entire file
    embedding = outputs.last_hidden_state[0, 0, :].numpy()
    return embedding


def main():
    print("Loading assignments and raw data...")
    try:
        # Load the file-to-microservice mapping from Experiment 3
        df_assignments = pd.read_csv(ASSIGNMENTS_FILE)

        # Load the raw extracted tokens
        with open(JSON_DATA_FILE, 'r') as f:
            raw_json = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find required file. {e}")
        return

    # Map filename to its raw tokens (reconstructed as a space-separated string)
    file_contents = {}
    for entry in raw_json:
        filename = entry.get('fileName')
        tokens = entry.get('tokens', [])
        # CodeBERT analyzes sequence and context, so we pass it the extracted tokens
        file_contents[filename] = " ".join(tokens)

    print(f"\nDownloading/Loading Pre-trained CodeBERT ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    print("\n--- PHASE 1: GENERATING CODEBERT EMBEDDINGS ---")
    file_embeddings = {}
    for filename, content in file_contents.items():
        if not content.strip():
            continue
        # Generate the embedding vector
        file_embeddings[filename] = get_codebert_embedding(content, tokenizer, model)
    print("Successfully generated embeddings for all files.")

    print("\n--- PHASE 2: CALCULATING SEMANTIC COHESION ---")

    # Group files by their assigned candidate microservice
    domains = ["Owner_Domain", "Vet_Domain", "Pet_Visit_Domain"]

    print("\n" + "=" * 70)
    print(f"{'MICROSERVICE CANDIDATE':<25} | {'FILES':<6} | {'AVG COSINE SIMILARITY':<20}")
    print("=" * 70)

    for domain in domains:
        # Get all files assigned to this domain
        domain_files = df_assignments[df_assignments['Assigned_Microservice'] == domain]['FileName'].tolist()

        # Filter to only files we successfully generated embeddings for
        valid_files = [f for f in domain_files if f in file_embeddings]

        if len(valid_files) < 2:
            print(f"{domain:<25} | {len(valid_files):<6} | {'N/A (Needs >= 2 files)':<20}")
            continue

        # Create a matrix of embeddings for the files in this cluster
        matrix = np.array([file_embeddings[f] for f in valid_files])

        # Calculate the pairwise cosine similarity
        similarity_matrix = cosine_similarity(matrix)

        # Extract the upper triangle of the matrix to get unique pairs (excluding self-similarity of 1.0)
        upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
        unique_similarities = similarity_matrix[upper_triangle_indices]

        # Calculate the average semantic cohesion
        avg_cohesion = np.mean(unique_similarities)

        print(f"{domain:<25} | {len(valid_files):<6} | {avg_cohesion:.4f}")

    print("=" * 70)



if __name__ == "__main__":
    main()