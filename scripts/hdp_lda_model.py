import pickle
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd

# --- CONFIGURATION (Toggle this between 'a' and 'b') ---
RUN_VERSION = 'b'  # CHANGE THIS TO 'b' WHEN RUNNING EXPERIMENT 2

DICTIONARY_FILE = f'dictionary_{RUN_VERSION}.gensim'
CORPUS_FILE = f'corpus_{RUN_VERSION}.pkl'
OUTPUT_MODEL = f'hdp_tuned_{RUN_VERSION}.model'

def main():
    print(f"Loading data for Corpus {RUN_VERSION.upper()}...")
    try:
        dictionary = corpora.Dictionary.load(DICTIONARY_FILE)
        with open(CORPUS_FILE, 'rb') as f:
            corpus = pickle.load(f)
    except FileNotFoundError:
        print(f" Error: Files for {RUN_VERSION} not found. Run convert_to_corpus.py first.")
        return

    print(f"\n--- EXPERIMENT TASK: TUNING HDP-LDA (Corpus {RUN_VERSION.upper()}) ---")

    # TUNING PARAMETERS (The "Address Limitations" part)
    hdp_tuned = models.HdpModel(
        corpus=corpus,
        id2word=dictionary,
        T=50,  # Force a hard limit of 50 topics maximum
        gamma=1.0,
        alpha=0.1,
        random_state=42
    )

    # CALCULATE COHERENCE SCORE 
    print("\nCalculating Coherence Score...")
    cm = CoherenceModel(model=hdp_tuned, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence_score = cm.get_coherence()
    print(f" U_Mass Coherence Score: {coherence_score:.4f}")

    # FILTERING WEAK TOPICS
    all_topics = hdp_tuned.print_topics(num_topics=50, num_words=5)
    meaningful_topics = []

    print("\n--- LABELED TOPICS (Candidate Microservices) ---")
    print(f"{'ID':<5} | {'Keywords (The Fingerprint)':<50} | {'Suggested Label'}")
    print("-" * 80)

    for topic_id, topic_string in all_topics:
        # ROBUST PARSING LOGIC
        words = []
        for w in topic_string.split(' + '):
            if '*' in w:
                clean_word = w.split('*')[1].strip().replace('"', '')
                words.append(clean_word)

        # Simple heuristic for labeling
        label = "Unknown"
        if any(x in words for x in ["pet", "visit"]):
            label = "Pet/Visit Domain"
        elif any(x in words for x in ["owner", "address", "city"]):
            label = "Owner Domain"
        elif any(x in words for x in ["vet", "specialty"]):
            label = "Vet Domain"
        elif any(x in words for x in ["user", "role", "login"]):
            label = "Auth/User Domain"
        else:
            label = "Infra/Abstraction"

        print(f"{topic_id:<5} | {', '.join(words):<50} | {label}")
        meaningful_topics.append({"ID": topic_id, "Words": words, "Label": label})

    print("-" * 80)
    print(f"\n Optimization Complete. Model produced {len(meaningful_topics)} active topics.")

    # Save the tuned model
    hdp_tuned.save(OUTPUT_MODEL)
    print(f"Saved tuned model to '{OUTPUT_MODEL}'")

if __name__ == "__main__":
    main()