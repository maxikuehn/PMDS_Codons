<div align="center">

<h1>Synonyme Codon Prediction</h1>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ddd.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMjgiIGhlaWdodD0iMTI4IiBzdHJva2U9IiM3NzciIGZpbGwtb3BhY2l0eT0iLjgiPgo8cGF0aCBmaWxsPSIjRkZGIiBkPSJtNjMsMWE2Myw2MyAwIDEsMCAyLDB6bTAsMTRhNDksNDkgMCAxLDAgMiwwem0wLDE0YTM1LDM1IDAgMSwwCjIsMHptMCwxNGEyMSwyMSAwIDEsMCAyLDB6bTAsMTRhNyw3IDAgMSwwIDIsMHptNjQsN0gxbTEwOC00NS05MCw5MG05MCwwLTkwLTkwbTQ1LTE4djEyNiIvPgo8cGF0aCBmaWxsPSIjRjYwIiBkPSJtNTAsOC0yMCwxMCA2OCw5MiAxMC0xMEw2NCw2NHoiLz4KPHBhdGggZmlsbD0iI0ZDMCIgZD0ibTE3LDUwdjI4TDY0LDY0eiIvPgo8cGF0aCBmaWxsPSIjN0Y3IiBkPSJtNjQsNjQgNiwzNUg1OHoiLz4KPHBhdGggZmlsbD0iI0NGMyIgZD0ibTY0LDY0IDEzLTQwIDksNXoiLz4KPHBhdGggZmlsbD0iIzA0RiIgZD0ibTY0LDY0IDE0LTYgMSw0emwtMjYsMTMgMyw0eiIvPgo8L3N2Zz4=&logoColor=black)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Gitea](https://img.shields.io/badge/Gitea-34495E?style=for-the-badge&logo=gitea&logoColor=5D9425)

<h4>Project Medical Data Science - SS 2024</h4>

*Insa Belter, Maximilian Kühn, Felix Mucha, Nils Rekus, Floris Wittner*
</div>

## Installation

### Requirements
- **Python 3.8 or higher** (Python 3.12 is recommended). More information on how to install Python can be found [here](https://www.python.org/downloads/).
- **pip** Python package installer (usually included in Python installations)

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
4. **Windows only:** to use Cuda for NVIDIA GPU acceleration, install pytorch with the following command first (for more information see [Pytorch Installation Guide](https://pytorch.org/get-started/locally/))
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
5. Install all other requirements
    ```bash
    pip install -r requirements.txt
    ``` 

## Project Content

### Folder Structure

```text 
.
├── data
│   └── organism
│       └── cleanedData.pkl
├── ml_models
│   └── organism
│       ├── best_rnn_model.pt
│       ├── best_tcn_model.pt
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
    - [07_1_TCN_Training](notebooks/07_1_TCN_Training.ipynb)
    - [07_2_TCN_Testing](notebooks/07_2_TCN_Testing.ipynb)
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


### Scripts
- Data Aggregation
    - [data loading](scripts/clean_and_pickle.py): Load Fasta files, check for corrupted sequences and save cleaned data
    - [data splitting](scripts/data_splitter.py): Data splitting for training, testing and validation
    - [codon usage bias](scripts/usage_bias_and_pickle.py): Calculation of Codon Usage Bias (CUB)
- [ML helper](scripts/ml_helper.py): Helper functions for training the machine learning models
- [ML evaluation](scripts/ml_evaluation.py): Evaluation functions for the machine learning models
- Files with classifier implementations for each model architecture
    - [Classifier Class](scripts/Classifier.py)
    - [Baseline](scripts/Baseline_classifiers.py)
    - [RNN](scripts/rnn.py)
    - [TCN](scripts/Tcnn.py)
    - [Encoder](scripts/encoder.py) & [Modified Pytorch Encoder](scripts/custom_transformer_encoder.py) (can output Attention weights)
    - [Index Classifier](scripts/index_classifier.py)
- [Chemical Property](scripts/chemicalProperty.py): Chemical property analysis of codons
- change Table
- [WIP] [secondary structure](scripts/secondary_structur.py): Exploration of secondary structure as additional feature


### Data
Contains the cleaned sequence data for 3 organisms (E.Coli, D.Melanogaster, H.Sapiens). The data is split in training, testing and validation data and then saved as a pickle file.
Also all the splits are saved in a shuffled version.

The data folder also contains various files, which track the model training progress and the model performance.