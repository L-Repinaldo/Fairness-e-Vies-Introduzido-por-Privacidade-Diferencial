from data import load_data
from model import run_logistic_regression
from experiments import run_experimemnt

import pandas as pd


if __name__ == '__main__':

    datasets = load_data()
    names = ["baseline", "eps_0.1", "eps_0.5", "eps_1.0", "eps_2.0"]

    experiments = [
        ("Logistic Regression" , run_logistic_regression),
    ]

    classification_rows = []
    data_rows = []

    for model_name, runner in experiments:

        print(f"\n{'='*40}")
        print(f"{model_name} execution")
        print(f"{'='*40}")

        data_output, model_output, cargo_output, setor_output = run_experimemnt(
            model_name= model_name,
            model_runner= runner,
            datasets=datasets,
            dataset_names=names
        )
        
        for dataset_name, payload in model_output.items():

            classification_rows.append({
                "model" : model_name,
                "dataset" : dataset_name,
                "true positive": payload['results']['tp'],
                "true negative": payload['results']['tn'],
                "false positive": payload['results']['fp'],
                "false negative": payload['results']['fn'],
                "true positive rate": payload['results']['tpr'],
                "false positive rate": payload['results']['fpr'],

            })
        
        for dataset_name, payload in data_output.items():

            data_rows.append({
                "dataset" : dataset_name,
                "mean": payload['mean'],
                "median": payload['median'],
                "std": payload['std'],
            })

        df_classification = pd.DataFrame(classification_rows)
        df_data = pd.DataFrame(data_rows)

        cargo_rows = []
        for dataset_name, payload in cargo_output.items():
            for row in payload["results"]:
                cargo_rows.append(row)

        setor_rows = []
        for dataset_name, payload in setor_output.items():
            for row in payload["results"]:
                setor_rows.append(row)

        df_cargo = pd.DataFrame(cargo_rows)
        df_setor = pd.DataFrame(setor_rows)

        print(df_classification)
        print()
        print(df_data)
        print()
        print(df_cargo)
        print()
        print(df_setor)
