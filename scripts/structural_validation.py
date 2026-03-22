import json
import pickle
import pandas as pd
from gensim import models

# --- CONFIGURATION ---
MODEL_FILE = '../data/hdp_tuned_b.model'
CORPUS_FILE = '../data/corpus_b.pkl'
FILENAMES_FILE = '../data/filenames_b.pkl'
JSON_DATA_FILE = '../data/petclinic_data.json'


def get_domain_label(topic_string):
    words = []
    for w in topic_string.split(' + '):
        if '*' in w:
            clean_word = w.split('*')[1].strip().replace('"', '')
            words.append(clean_word)

    if any(x in words for x in ["pet", "visit"]):
        return "Pet_Visit_Domain"
    elif any(x in words for x in ["owner", "address", "city"]):
        return "Owner_Domain"
    elif any(x in words for x in ["vet", "specialty"]):
        return "Vet_Domain"
    else:
        return "Infra_Abstraction"


def extract_class_name(method_call_string):
    if '.' in method_call_string:
        class_name = method_call_string.split('.')[0]
        return f"{class_name}.java"
    return None


def main():
    print("Loading AI Models and Data...")
    try:
        hdp_model = models.HdpModel.load(MODEL_FILE)
        with open(CORPUS_FILE, 'rb') as f:
            corpus = pickle.load(f)
        with open(FILENAMES_FILE, 'rb') as f:
            filenames = pickle.load(f)
        with open(JSON_DATA_FILE, 'r') as f:
            raw_json = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    print("\n--- PHASE 1: ASSIGNING FILES TO MICROSERVICES ---")
    file_to_domain = {}

    for i, (doc_bow, filename) in enumerate(zip(corpus, filenames)):
        topic_probs = hdp_model[doc_bow]

        if not topic_probs:
            file_to_domain[filename] = "Infra_Abstraction"
            continue

        # Sort by highest probability to find the dominant topic
        topic_probs.sort(key=lambda x: x[1], reverse=True)
        dominant_topic_id = topic_probs[0][0]

        # Get the words for this topic and apply your label heuristic
        topic_words = hdp_model.print_topic(dominant_topic_id)
        domain_label = get_domain_label(topic_words)

        file_to_domain[filename] = domain_label


    print("Files assigned to semantic domains.")
    df_assignments = pd.DataFrame(list(file_to_domain.items()), columns=['FileName', 'Assigned_Microservice'])
    df_assignments.to_csv('microservice_assignments.csv', index=False)
    print("Assignments file 'microservice_assignments.csv' has been created.")
    print("\n--- PHASE 2: CALCULATING MODULARITY SCORES (Ms) ---")

    # Initialize metrics for each microservice candidate
    metrics = {
        "Owner_Domain": {"internal": 0, "external": 0},
        "Vet_Domain": {"internal": 0, "external": 0},
        "Pet_Visit_Domain": {"internal": 0, "external": 0}
    }

    # Extract call graph from JSON
    for entry in raw_json:
        caller_file = entry.get('fileName')
        called_methods = entry.get('methodCalls', [])

        caller_domain = file_to_domain.get(caller_file)

        # We only calculate metrics for files inside our 3 valid business domains
        if caller_domain not in metrics:
            continue

        for method_call in called_methods:
            callee_file = extract_class_name(method_call)

            if not callee_file:
                continue

            callee_domain = file_to_domain.get(callee_file)

            # 1. Ignore calls to external Java/Spring libraries (e.g., 'Objects.equals')
            # 2. Ignore calls to technical/infra files in our own project
            if not callee_domain or callee_domain == "Infra_Abstraction":
                continue

            if caller_domain == callee_domain:
                metrics[caller_domain]["internal"] += 1
            else:
                metrics[caller_domain]["external"] += 1

    # Print Results Table
    print("\n" + "=" * 60)
    print(f"{'MICROSERVICE CANDIDATE':<25} | {'INTERNAL':<8} | {'EXTERNAL':<8} | {'M_s SCORE':<10}")
    print("=" * 60)

    for domain, counts in metrics.items():
        internal = counts["internal"]
        external = counts["external"]
        total = internal + external

        if total == 0:
            ms_score = 0.0
            print(f"{domain:<25} | {internal:<8} | {external:<8} | {'N/A':<10}")
        else:
            ms_score = internal / total
            print(f"{domain:<25} | {internal:<8} | {external:<8} | {ms_score:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()