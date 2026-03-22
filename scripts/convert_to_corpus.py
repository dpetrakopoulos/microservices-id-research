import json
import pickle
import nltk
from gensim import corpora
from nltk.corpus import stopwords

# Download standard English stop words (run once)
nltk.download('stopwords')

# --- CONFIGURATION ---
INPUT_FILE = '../data/petclinic_data.json'
OUTPUT_DICTIONARY = '../data/dictionary_b.gensim'
OUTPUT_CORPUS = '../data/corpus_b.pkl'
OUTPUT_FILENAMES = '../data/filenames_b.pkl'

#Corpus A
OUTPUT_DICTIONARY = 'dictionary_b.gensim'  # Changed to _a
OUTPUT_CORPUS = 'corpus_b.pkl'             # Changed to _a
OUTPUT_FILENAMES = 'filenames_b.pkl'

# 1. JAVA KEYWORDS (Keep these removed)
JAVA_KEYWORDS = {
    'public', 'private', 'protected', 'void', 'string', 'int', 'boolean',
    'class', 'interface', 'extends', 'implements', 'return', 'if', 'else',
    'for', 'while', 'do', 'try', 'catch', 'throw', 'new', 'package', 'import',
    'get', 'set', 'is', 'list', 'map', 'set', 'collection', 'object', 'byte',
    'short', 'long', 'float', 'double', 'char', 'final', 'static', 'this',
    'super', 'null', 'true', 'false', 'var', 'args', 'main', 'id', 'obj'
}

# 2. FRAMEWORK NOISE (Keep strict Spring MVC removal, but SAVE domain words)
# removed 'owners', 'date', 'type', 'name' from this list so they survive.
DOMAIN_NOISE = {
    'base', 'entity', 'model', 'dto', 'dao', 'repository', 'controller',
    'service', 'impl', 'util', 'helper', 'configuration', 'application',
    'system', 'support', 'named', 'factory', 'builder', 'formatter', 'validator',

    # Spring MVC keywords (These are safe to remove)
    'process', 'redirect', 'allowed', 'data', 'locale', 'update', 'init',
    'attributes', 'result', 'pagination', 'paginated', 'pageable', 'binder',
    'show', 'form', 'creation', 'fields', 'add', 'find', 'last'
}

STOP_WORDS = set(stopwords.words('english')).union(JAVA_KEYWORDS).union(DOMAIN_NOISE)


def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def clean_tokens(token_list):
    cleaned = []
    for token in token_list:
        token = token.lower()
        # --- COMMENT THESE OUT FOR CORPUS A ---
        if token in STOP_WORDS:
            continue
        if len(token) < 3:
            continue


        if not token.isalpha():
            continue
        cleaned.append(token)
    return cleaned


def main():
    print(f"Reading {INPUT_FILE}...")
    try:
        raw_data = load_data(INPUT_FILE)
    except FileNotFoundError:
        print(f" Error: Could not find '{INPUT_FILE}'.")
        return

    processed_docs = []
    doc_filenames = []

    for entry in raw_data:
        # Debug: Print the first few tokens of the first file to check if CamelCase is split
        if len(processed_docs) == 0:
            print(f"DEBUG: Raw tokens sample from {entry['fileName']}: {entry['tokens'][:10]}")

        tokens = clean_tokens(entry['tokens'])

        if tokens:
            processed_docs.append(tokens)
            doc_filenames.append(entry['fileName'])

    print(f"Processed {len(processed_docs)} valid documents.")

    # 1. Build Dictionary
    dictionary = corpora.Dictionary(processed_docs)

    # --- FIX: Changed no_below to 1 ---
    # We want to keep words even if they only appear in 1 file
    # (crucial for small datasets like PetClinic)
    dictionary.filter_extremes(no_below=1, no_above=1.0)

    print(f"Dictionary contains {len(dictionary)} unique domain terms.")

    # Debug: Print top 20 words to verify they are real domain terms
    print("DEBUG: Sample Vocabulary:", list(dictionary.token2id.keys())[:20])

    # 2. Build Corpus
    corpus = [dictionary.doc2bow(text) for text in processed_docs]

    # 3. Save Files
    dictionary.save(OUTPUT_DICTIONARY)
    with open(OUTPUT_CORPUS, 'wb') as f:
        pickle.dump(corpus, f)
    with open(OUTPUT_FILENAMES, 'wb') as f:
        pickle.dump(doc_filenames, f)

    print("\n Success! Files generated.")


if __name__ == "__main__":
    main()