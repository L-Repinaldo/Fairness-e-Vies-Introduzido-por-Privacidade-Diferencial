from statistics import mean
import pandas as pd

from metrics import compute_data_metrics, compute_model_metrics
from . import run_model, model_metrics

SEEDS = [42, 123, 2026]
TEST_SIZES = [0.2, 0.3]

MIN_GROUP_SIZE = 30 


def _aggregate_metrics(metrics_list):
    return {
        k: round(mean(m[k] for m in metrics_list), 3)
        for k in metrics_list[0].keys()
    }


def _add_salary_class(df, stats):
    upper = stats["mean"] + stats["std"]

    df_out = df.copy()
    df_out["salario_classe"] = (df_out["salario"] > upper).astype(int)

    return df_out


def _aggregate_group_metrics(metrics_list):
    if not metrics_list:
        return {}

    total_support = sum(m["support"] for m in metrics_list)

    def weighted_avg(key):
        return sum(m[key] * m["support"] for m in metrics_list) / total_support

    return {
        "true positive": round(weighted_avg("tp")),
        "true negative": round(weighted_avg("tn")),
        "false positive": round(weighted_avg("fp")),
        "false negative": round(weighted_avg("fn")),
        "true positive rate": round(weighted_avg("tpr"), 3),
        "false positive rate": round(weighted_avg("fpr"), 3),
        "support": total_support
    }


def _group_metrics_for_run(df, y_true, y_pred, dataset_name, group_col):
    if group_col not in df.columns:
        raise ValueError(f"Coluna '{group_col}' não encontrada.")

    group_values = df.loc[y_true.index, group_col]
    y_pred_series = pd.Series(y_pred, index=y_true.index)

    rows = []

    for group in group_values.unique():
        mask = group_values == group
        support = int(mask.sum())

        if support < MIN_GROUP_SIZE:
            continue

        metrics = compute_model_metrics(
            y_true=y_true[mask],
            y_pred=y_pred_series[mask],
        )

        rows.append({
            "dataset": dataset_name,
            "grupo": group,
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tpr": metrics["tpr"],
            "fpr": metrics["fpr"],
            "support": support,
        })

    return rows


def run_experimemnt(model_name, model_runner, datasets, dataset_names):

    data_stats = {}
    model_experiment_output = {}
    experiment_setor_output = {}

    for name, df in zip(dataset_names, datasets):

        stats = compute_data_metrics(df['salario'])
        data_stats[name] = stats

        df_with_target = _add_salary_class(df, stats)

        setores = df["setor"].unique()
        setor_group_runs = {setor: [] for setor in setores}

        model_runs = []

        for seed in SEEDS:
            for test_size in TEST_SIZES:

                model_results = run_model(
                    df=df_with_target,
                    model_name=model_name,
                    model_runner=lambda **kwargs: model_runner(
                        **kwargs,
                        seed=seed,
                        test_size=test_size,
                        target="salario_classe"
                    )
                )

                setor_rows = _group_metrics_for_run(
                    df=df,
                    y_true=model_results["results"]["y_test_true"],
                    y_pred=model_results["results"]["y_test_pred"],
                    dataset_name=name,
                    group_col="setor",
                )

                for row in setor_rows:
                    setor_group_runs[row["grupo"]].append(row)

                model_metrics_values = model_metrics(
                    model_output=model_results,
                    dataset_name=name,
                    model_name=model_name
                )

                model_runs.append(model_metrics_values['results'])

        model_experiment_output[name] = {
            "model_name": model_name,
            "results": _aggregate_metrics(model_runs)
        }

        setor_results = []
        for setor, metrics_list in setor_group_runs.items():
            aggregated = _aggregate_group_metrics(metrics_list)

            if aggregated: 
                setor_results.append({
                    "dataset": name,
                    "grupo": setor,
                    **aggregated,
                })

        experiment_setor_output[name] = {
            "model_name": model_name,
            "results": setor_results
        }

    return data_stats, model_experiment_output, experiment_setor_output