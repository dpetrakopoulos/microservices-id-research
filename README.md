# Microservices Identification Research

This repository contains the experimental framework for identifying microservice boundaries within the **Spring PetClinic** monolithic application. The project uses a multi-stage pipeline: **Static Analysis (Java)** → **Topic Modeling (HDP-LDA)** → **Structural Validation ($M_s$)** → **Semantic Verification (CodeBERT)**.

## 📂 Repository Structure
* `extractor-tool/`: Java 11 Maven tool for AST parsing and call graph generation.
* `spring-petclinic/`: The subject monolithic system (Source: [Spring Framework GitHub](https://github.com/spring-projects/spring-petclinic)).
* `data/`: Contains `petclinic_data.json` and the serialized corpora (`.pkl`) for Experiments 1 & 2.
* `scripts/`: Python suite for HDP-LDA modeling and architectural validation.

---

## 🚀 Execution Guide

### **Step 1: Data Extraction (Java)**
Build and run the extractor to generate the structural and semantic foundation.
* **Requirements**: JDK 11, Maven.
* **Command**:
    ```bash
    cd extractor-tool
    mvn clean compile exec:java -Dexec.mainClass="PetClinicExtractor"
    ```
* **Output**: `data/petclinic_data.json`.

### **Step 2: Preprocessing (Python)**
Generate the baseline (Corpus A) and filtered (Corpus B) datasets.
* **Command**: `python scripts/convert_to_corpus.py`

### **Step 3: Topic Modeling (Experiments 1 & 2)**
Run the HDP-LDA model to discover latent domains and calculate $U_{Mass}$ Coherence.
* **Command**: `python scripts/hdp_lda_model.py`
* **Note**: Toggle `RUN_VERSION = 'a'` or `'b'` in the script to switch between experiments.

### **Step 4: Structural & Semantic Validation (Experiments 3 & 4)**
Validate the identified boundaries using Modularity Scores ($M_s$) and CodeBERT embeddings.
* **Structural**: `python scripts/structural_validation.py`
* **Semantic**: `python scripts/semantic_validation.py`

---

## 📊 Summary of Research Findings
| Experiment | Focus | Key Metric | Result |
| :--- | :--- | :--- | :--- |
| **Exp 1** | Baseline LDA | $U_{Mass}$ Coherence | -20.4936 |
| **Exp 2** | Abstraction Handling | Domain Purity | High |
| **Exp 3** | Structural integrity | Modularity ($M_s$) | 0.75 (Owner Domain) |
| **Exp 4** | Semantic context | Cosine Similarity | 0.99 (CodeBERT) |

---

