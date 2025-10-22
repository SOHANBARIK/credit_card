import pathlib
import yaml
import sys
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def split_data(df, test_split, seed):
    # Split the dataset into train and test sets
    train, test = train_test_split(df, test_size=test_split, random_state=seed)
    return train, test

def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    output = pathlib.Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output / 'train.csv', index=False)
    test.to_csv(output / 'test.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/test")
    parser.add_argument("input_file", help="Input CSV file path")
    parser.add_argument("output_file", help="Output CSV file path")
    args = parser.parse_args()

    curr_dir = pathlib.Path(__file__).resolve()
    home_dir = curr_dir.parents[2]  # repository root
    params_file = home_dir / 'params.yaml'

    # Load parameters with defaults
    params = {"test_split": 0.2, "seed": 42}
    if params_file.exists():
        try:
            loaded = yaml.safe_load(params_file.open())
            if isinstance(loaded, dict) and "make_dataset" in loaded:
                params.update(loaded["make_dataset"])
        except Exception as e:
            print(f"Warning: Could not load params.yaml: {e}")

    input_path = pathlib.Path(args.input_file)
    if not input_path.is_absolute():
        input_path = (home_dir / input_path).resolve()

    output_path = pathlib.Path(args.output_file).parent
    if not output_path.is_absolute():
        output_path = (home_dir / output_path).resolve()

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    data = load_data(input_path)
    train_data, test_data = split_data(data, params["test_split"], params["seed"])
    save_data(train_data, test_data, output_path)

if __name__ == "__main__":
    main()











# # make_dataset.py
# import pathlib
# import yaml
# import sys
# import pandas as pd
# from sklearn.model_selection import train_test_split

# def load_data(data_path):
#     # Load your dataset from a given path
#     df = pd.read_csv(data_path)
#     return df

# def split_data(df, test_split, seed):
#     # Split the dataset into train and test sets
#     train, test = train_test_split(df, test_size=test_split, random_state=seed)
#     return train, test

# def save_data(train, test, output_path):
#     # Save the split datasets to the specified output path
#     pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
#     train.to_csv(output_path + '/train.csv', index=False)
#     test.to_csv(output_path + '/test.csv', index=False)

# def main():

#     curr_dir = pathlib.Path(__file__)
#     home_dir = curr_dir.parent.parent.parent
#     params_file = home_dir.as_posix() + '/params.yaml'
#     params = yaml.safe_load(open(params_file))["make_dataset"]

#     input_file = sys.argv[1]
#     data_path = home_dir.as_posix() + input_file
#     output_path = home_dir.as_posix() + '/data/processed'
    
#     data = load_data(data_path)
#     train_data, test_data = split_data(data, params['test_split'], params['seed'])
#     save_data(train_data, test_data, output_path)

# if __name__ == "__main__":
#     main()