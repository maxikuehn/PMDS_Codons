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
4. Install the requirements
    ```bash
    pip install -r requirements.txt
    ``` 
5. to use Cuda, install pytorch with the following command
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

[TODO] prüfen, ob das so klappt

## Folder Structure
...

### Notebooks
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
    - ...
- 06: RNN -> Maxi
    - 061_RNN_Training
    - 062_RNN_Testing
- 07: TCNN -> Felix
    - 071_TCNN_Training
    - 072_TCNN_Testing
- 08: Encoder-only Transformer -> Insa
    - 081_Encoder_Training
    - 082_Encoder_Testing
- 09: Accuracy Results Overview -> Insa
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