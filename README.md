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


## Folder Structure

```text 
.
├── data
│   └── organism
│       └── cleanedData.pkl
├── ml_models
│   └── organism
│       ├── best_rnn_model
│       ├── best_tcn_model
│       └── best_encoder_model.pt
├── notebooks
│   ├── archive
│   └── *.ipynb
├── scripts
│   ├── archive
│   └── *.py
├── unit_tests
├── README.md
└── requirements.txt
```
    
### Notebooks
- [00_Using_Classifiers](notebooks/00_Using_Classifiers.ipynb)
    - Notebook for applying the best classifiers from each model architecture to new data
- [01_Data_Aggregation_and_Preparation](notebooks/01_Data_Aggregation_and_Preparation.ipynb)
    - Data aggregation: How was the data gathered?
    - Data preparation: How was the data cleaned and split for training and testing purposes
    - Secondary Protein Structure as possible additional feature
- [02_Data_Exploration](notebooks/02_Data_Exploration.ipynb)
    - How many sequences do we have per organism? 
    - How are the amino acids distributed?
    - What is the Codon Usage Bias? 
- [03_Baseline_Classifiers](notebooks/03_Baseline_Classifiers.ipynb)
    - Classifiers for comparing the trained machine learning classifiers to the possible baseline resulting from the Codon Usage Bias (CUB)
- [04_Index_based_Analysis_and_Classifier](notebooks/04_Index_based_Analysis_and_Classifier.ipynb)
    - Index based analysis of amino acid sequences
    - Classifier based on results of this analysis (Index based CUB)
- [05_Other_Statistical_Analysis_Approaches](notebooks/05_Other_Statistical_Analysis_Approaches.ipynb)
    - Correlation between neighbouring amino acids
    - Amino acid analysis based of chemical properties of codons
- 06: RNN
    - [06_1_RNN_Training](notebooks/06_1_RNN_Training.ipynb)
    - [06_2_RNN_Testing](notebooks/06_2_RNN_Testing.ipynb)
- 07: TCNN
    - [07_1_TCNN_Training](notebooks/07_1_TCNN_Training.ipynb)
    - [07_2_TCNN_Testing](notebooks/07_2_TCNN_Testing.ipynb)
- 08: Encoder-only Transformer
    - [08_1_Encoder_Training](notebooks/08_1_Encoder_Training.ipynb)
    - [08_2_Encoder_Testing](notebooks/08_2_Encoder_Testing.ipynb)
- [09_Accuracy_Results_Overview](notebooks/09_Accuracy_Results_Overview.ipynb)
    - Training validation accuracies per model (RNN, Encoder, TCNN)
    - Accuracy per (best) Model per Organism in comparison to baseline 
        - Index-based CUB vs Baseline Max CUB
        - RNN vs Baseline Max CUB
        - TCNN vs Baseline Max CUB
        - Encoder vs Baseline Max CUB
    - All accuracies in one diagram
        - Max CUB, Index-based Max CUB, RNN, TCNN, Transformer

<!-- #### Testing Notebooks Content
Per Organism:
- Index-based Segment Accuracy
- Confusion Matrix (Codons)
- Confusion Matrix (Amino acids)
- Codon Usage Bias
- Accuracy per Codon
- Relative Prediction Frequency per Codon
- Comparison Max CUB with Model Diagrams -->


### Scripts


### Data


### Machine Learning Models