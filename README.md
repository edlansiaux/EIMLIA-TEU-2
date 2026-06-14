# EIMLIA — Explainable AI for Emergency Department Triage and Simulation

## Overview

**EIMLIA** is an end-to-end research framework designed to evaluate, optimize, and simulate emergency department (ED) triage processes using artificial intelligence, process mining, and hybrid healthcare simulation.

The project combines:

- Natural Language Processing (NLP) for clinical triage prediction
- Machine Learning and Deep Learning models
- FRENCH Triage V2 guideline-based target generation
- Process Mining using PM4Py
- Hybrid Simulation (Discrete Event Simulation + Agent-Based Modeling)
- Health-economic evaluation (ROI, ICER, TCO)

The objective is to assess whether AI-assisted triage can improve patient flow, reduce waiting times, decrease overcrowding, and increase triage consistency in emergency departments.

---

## Project Workflow

### 1. Data Preparation

The notebook begins by:

- Importing raw emergency department datasets (2018–2023)
- Merging multiple CSV files into a unified database
- Cleaning and harmonizing variables
- Preparing structured and unstructured clinical data

**Output:**

```text
fusion_totale_2018_2023.csv
```

---

### 2. FRENCH Triage Ground Truth Generation

Instead of training models directly on nurse-assigned triage levels, EIMLIA computes an **ideal FRENCH triage level** using the official SFMU FRENCH Triage V2 algorithm.

This provides:

- Standardized labels
- Reduced inter-rater variability
- Reproducible target generation

The original nurse-assigned triage levels are preserved for comparison.

---

### 3. Exploratory Data Analysis

The notebook performs:

- Feature distribution analysis
- Missing data inspection
- CCMU transformation
- Triage-level distribution analysis
- Correlation analysis
- Visual exploratory statistics

---

## AI Models

### TRIAGEMASTER

**Architecture**

- Doc2Vec text embeddings
- Multi-Layer Perceptron (MLP)
- SHAP explainability

**Purpose**

Baseline explainable NLP triage model.

---

### URGENTIAPARSE

**Architecture**

- FlauBERT transformer embeddings
- XGBoost classifier

**Purpose**

Advanced French-language clinical text understanding.

---

### EMERGINET

**Architecture**

- JEPA-inspired representation learning
- VICReg self-supervised regularization
- Deep neural classifier

**Purpose**

State-of-the-art triage prediction framework.

---

## Training Pipeline

The dataset is split into:

```text
Train:      64%
Validation: 16%
Test:       20%
```

Training includes:

- Stratified sampling
- Early stopping
- Model persistence
- Comparative evaluation

### Evaluation Metrics

- Accuracy
- Weighted F1-score
- Mean Absolute Error (MAE)
- Cohen's Kappa
- Confusion matrices

---

## Process Mining

The framework implements a complete process mining workflow using **PM4Py**.

### Step 1 – Event Log Generation

Synthetic event logs are generated from emergency department workflows.

### Step 2 – Process Discovery

Inductive Miner is used to discover actual patient-care pathways.

### Step 3 – Conformance Checking

Evaluation of process quality through:

- Token Replay
- Precision
- Fitness
- Generalization

### Step 4 – KPI Calibration

Empirical distributions are extracted and calibrated for simulation.

### Step 5 – Ecological Validation

Simulation outputs are compared against real hospital data.

---

## Hybrid Simulation Framework

EIMLIA integrates two complementary paradigms.

### Discrete Event Simulation (DES)

Implemented using SimPy.

Models:

- Patient arrivals
- Waiting queues
- Resource utilization
- Throughput

### Agent-Based Modeling (ABM)

Implemented using Mesa.

Models:

- Clinician behavior
- Decision variability
- Human-system interactions

### Hybrid DES + ABM

Combines operational flow and human behavior within a single simulation environment.

---

## Scenario Analysis

Several organizational scenarios can be evaluated:

- Current practice (baseline)
- TRIAGEMASTER-assisted triage
- URGENTIAPARSE-assisted triage
- EMERGINET-assisted triage
- Hybrid deployment strategies

Stress-testing scenarios include:

- Increased patient volume
- Staffing shortages
- Epidemic surges
- Resource constraints

---

## Economic Evaluation

### Cost Analysis

- Implementation costs
- Infrastructure costs
- Maintenance costs

### Outcome Analysis

- Reduced waiting times
- Reduced overcrowding
- Improved throughput

### Economic Metrics

- Return on Investment (ROI)
- Incremental Cost-Effectiveness Ratio (ICER)
- Total Cost of Ownership (TCO)

---

## Statistical Validation

Comparative analyses include:

- Bootstrap confidence intervals
- Cohen's κ estimation
- Z-tests
- Multiple-comparison correction (Bonferroni)

These procedures ensure robust comparison between AI-assisted and standard triage workflows.

---

## Expected Outputs

The notebook produces:

- Trained AI models
- Triage predictions
- Comparative performance reports
- Process mining models
- Simulation results
- Scenario comparison dashboards
- Economic evaluation reports
- Statistical validation summaries

---

## Technology Stack

### Data Science

- Python
- Pandas
- NumPy
- SciPy

### Machine Learning

- Scikit-learn
- XGBoost
- Doc2Vec
- FlauBERT

### Explainability

- SHAP

### Process Mining

- PM4Py

### Simulation

- SimPy
- Mesa

### Visualization

- Matplotlib
- Seaborn

---

## Research Objective

EIMLIA aims to provide a reproducible framework for studying how explainable artificial intelligence, process mining, and hybrid simulation can support emergency department decision-making, improve triage reliability, and evaluate organizational interventions before real-world deployment.

---

## License

This project is intended for academic and research purposes. Clinical deployment requires external validation, regulatory review, and compliance with applicable healthcare regulations.
