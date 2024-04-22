import pandas as pd
import os


def get_cleaned_data(data_dir):
    # get pkl files
    pkl_files = {}
    for root, dirs, _ in os.walk(data_dir):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            for file in os.listdir(dir_path):
                if file.endswith("cleanedData.pkl"):
                    pkl_files[dir] = os.path.join(dir_path, file)
    
    return pkl_files


# calculate usage bias with absolute values
def calculate_usage_bias_absolut(pkl_file):
    usage_bias = {}
    data = pd.read_pickle(pkl_file)
    # zip sequence in 3 positions with translations
    ziped = zip(data['sequence'].apply(lambda x: [str(x[i:i+3]) for i in range(0, len(x), 3)]), data['translation'])

    # iterate over zipped data and count codon usage
    for sequence, translation in ziped:
        for codon, aa in zip(sequence, translation):
            if aa not in usage_bias:
                usage_bias[aa] = {}
            if codon not in usage_bias[aa]:
                usage_bias[aa][codon] = 0
            usage_bias[aa][codon] += 1

    return usage_bias

def calculate_usage_bias_relative(pkl_file):
    usage_bias = calculate_usage_bias_absolut(pkl_file)
    usage_bias_relative = {}
    for aa in usage_bias:
        total = sum(usage_bias[aa].values())
        usage_bias_relative[aa] = {k: v/total for k, v in usage_bias[aa].items()}
    return usage_bias_relative

def main(dir_data, dir_output=None, organism=None):

    pkl_files = get_cleaned_data(dir_data)
    
    if organism:
        pkl_files = {organism: pkl_files[organism]}

    for file in pkl_files:
        print(f"Calculating usage bias for: {file}")
        usage_bias = calculate_usage_bias_relative(pkl_files[file])

        output_dir = dir_output if dir_output else os.path.dirname(pkl_files[file])
        
        print(f"usage bias will be saved in: {output_dir}")
        pd.to_pickle(usage_bias, os.path.join(output_dir, f"usageBias.pkl"))


if __name__ == "__main__":
    main("./data/", dir_output=None, organism=None)


