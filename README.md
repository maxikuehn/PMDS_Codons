# Optimale Synonyme Codons

## Installation

### Requirements
- Python 3.8 or higher
- pip

### Setup
1. Clone the repository
2. Create a virtual environment
    ```bash
    python -m venv .venv
    ```
3. Activate the virtual environment
    ```bash
    source .venv/bin/activate
    ```
4. to use Cuda for NVIDIA GPU acceleration, install pytorch with the following command first
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
5. Install all other requirements
    ```bash
    pip install -r requirements.txt
    ``` 

[TODO] prüfen, ob das so klappt

## Folder Structure
...

### Notebooks
- 00_Using_Classifiers
    - Notebook for applying the best classifiers from each model architecture to new data
- 01_Data_Aggregation_and_Preparation -> Floris
    - Data aggregation: How was the data gathered?
    - Data preparation: How was the data cleaned and split for training and testing purposes
    - Secondary Protein Structure as possible additional feature
- 02_Data_Exploration -> Maxi
    - How many sequences do we have per organism? 
    - How are the amino acids distributed?
    - What is the Codon Usage Bias? 
- 03_Baseline_Classifiers -> Insa
    - Classifiers for comparing the trained machine learning classifiers to the possible baseline resulting from the Codon Usage Bias (CUB)
- 04_Index_based_Analysis_and_Classifier -> Nils
    - Index based analysis of amino acid sequences
    - Classifier based on results of this analysis (Index based CUB)
- 05_Other_Statistical_Analysis_Approaches -> Nils, Floris
    - Correlation between neighbouring amino acids
    - Amino acid analysis based of chemical properties of codons
- 06: RNN -> Maxi
    - 06_1_RNN_Training
    - 06_2_RNN_Testing
- 07: TCNN -> Felix
    - 07_1_TCNN_Training
    - 07_2_TCNN_Testing
- 08: Encoder-only Transformer -> Insa
    - 08_1_Encoder_Training
    - 08_2_Encoder_Testing
- 09: Accuracy Results Overview -> Insa
    - Training validation accuracies per model (RNN, Encoder, TCNN)
    - Accuracy per (best) Model per Organism in comparison to baseline 
        - Index-based CUB vs Baseline Max CUB
        - RNN vs Baseline Max CUB
        - TCNN vs Baseline Max CUB
        - Encoder vs Baseline Max CUB
    - All accuracies in one diagram
        - Max CUB, Index-based Max CUB, RNN, TCNN, Transformer

#### Testing Notebooks Content
Per Organism:
- Index-based Segment Accuracy
- Confusion Matrix (Codons)
- Confusion Matrix (Amino acids)
- Codon Usage Bias
- Accuracy per Codon
- Relative Prediction Frequency per Codon
- Comparison Max CUB with Model Diagrams


### Scripts


### Data


### Machine Learning Models