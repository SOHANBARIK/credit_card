# # ...existing code...
# import argparse
# import pathlib
# import sys
# import yaml
# import joblib
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier

# def train_model(train_features: pd.DataFrame, target: pd.Series, n_estimators: int, max_depth, seed: int):
#     """Train and return a RandomForestClassifier."""
#     model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
#     model.fit(train_features, target)
#     return model

# def save_model(model, output_path: str):
#     """Ensure output directory exists and save model.joblib inside it."""
#     outp = pathlib.Path(output_path)
#     outp.mkdir(parents=True, exist_ok=True)
#     joblib.dump(model, outp / "model.joblib")

# def main():
#     parser = argparse.ArgumentParser(description="Train model")
#     parser.add_argument("input_dir", nargs="?", default="data/processed", help="Path to processed data folder")
#     parser.add_argument("--output-dir", default="models", help="Path to save model")
#     args = parser.parse_args()

#     curr_dir = pathlib.Path(__file__).resolve()
#     repo_root = curr_dir.parents[2]  # .../creditcard
#     params_file = repo_root / "params.yaml"

#     # Load params with safe defaults
#     params = {"n_estimators": 100, "max_depth": None, "seed": 42}
#     if params_file.exists():
#         try:
#             loaded = yaml.safe_load(params_file.open())
#             if isinstance(loaded, dict) and "train_model" in loaded:
#                 params.update(loaded["train_model"])
#         except Exception:
#             pass

#     # Resolve input path (handles .\data\processed\ from dvc.yaml)
#     input_path = pathlib.Path(args.input_dir)
#     if not input_path.is_absolute():
#         data_path = (repo_root / input_path).resolve()
#     else:
#         data_path = input_path.resolve()

#     # Resolve and create output path
#     output_path = (repo_root / pathlib.Path(args.output_dir)).resolve()
#     output_path.mkdir(parents=True, exist_ok=True)

#     TARGET = "Class"
#     train_csv = data_path / "train.csv"
#     if not train_csv.exists():
#         print(f"Error: expected train.csv in {data_path!s}")
#         sys.exit(1)

#     df = pd.read_csv(train_csv)
#     if TARGET not in df.columns:
#         print(f"Error: target column '{TARGET}' not found in {train_csv}")
#         sys.exit(1)

#     X = df.drop(TARGET, axis=1)
#     y = df[TARGET]

#     trained_model = train_model(X, y, int(params.get("n_estimators", 100)), params.get("max_depth"), int(params.get("seed", 42)))
#     save_model(trained_model, str(output_path))

# if __name__ == "__main__":
#     main()
# # ...existing code...


# # ...existing code...
# import argparse
# # ...existing code...

# def save_model(model, output_path):
#     # Save the trained model to the specified output path
#     joblib.dump(model, str(pathlib.Path(output_path) / "model.joblib"))

# def main():
#     parser = argparse.ArgumentParser(description="Train model")
#     parser.add_argument("input_dir", nargs="?", default="data/processed", help="Path to processed data folder")
#     parser.add_argument("--output-dir", default="models", help="Path to save model")
#     args = parser.parse_args()

#     curr_dir = pathlib.Path(__file__).resolve()
#     repo_root = curr_dir.parents[2]  # repository root: .../creditcard
#     params_file = repo_root / "params.yaml"
#     params = yaml.safe_load(params_file.open())["train_model"]

#     input_path = pathlib.Path(args.input_dir)
#     if not input_path.is_absolute():
#         data_path = (repo_root / input_path).resolve()
#     else:
#         data_path = input_path.resolve()

#     output_path = (repo_root / pathlib.Path(args.output_dir)).resolve()
#     output_path.mkdir(parents=True, exist_ok=True)

#     TARGET = "Class"
#     train_csv = data_path / "train.csv"
#     if not train_csv.exists():
#         print(f"Error: expected train.csv in {data_path!s}")
#         sys.exit(1)

#     df = pd.read_csv(train_csv)
#     X = df.drop(TARGET, axis=1)
#     y = df[TARGET]

#     trained_model = train_model(X, y, params["n_estimators"], params["max_depth"], params["seed"])
#     save_model(trained_model, str(output_path))

# # ...existing code...






# train_model.py
import pathlib
import sys
import yaml
import joblib

import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_model(train_features, target, n_estimators, max_depth, seed):
    # Train your machine learning model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    model.fit(train_features, target)
    return model

def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + '/model.joblib')

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["train_model"]

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    TARGET = 'Class'
    train_features = pd.read_csv(data_path + '/train.csv')
    X = train_features.drop(TARGET, axis=1)
    y = train_features[TARGET]

    trained_model = train_model(X, y, params['n_estimators'], params['max_depth'], params['seed'])
    save_model(trained_model, output_path)

    

if __name__ == "__main__":
    main()
