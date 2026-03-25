from statistics import mean, stdev
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import load_data
from experiments import run_model
from experiments.experiments_runner import _add_salary_class
from metrics import compute_model_metrics, compute_data_metrics
from model import run_logistic_regression
from preprocessing import build_preprocessor

SEEDS = [42, 123, 2026]
TEST_SIZE = 0.3

STABILITY_STD_THRESHOLD = 0.05
OVERFIT_GAP_THRESHOLD = 0.10
UNDERFIT_TPR_THRESHOLD = 0.55
IMBALANCE_THRESHOLD = 0.05


def _mean_std(values):
    if not values:
        return 0.0, 0.0

    if len(values) == 1:
        return values[0], 0.0

    return mean(values), stdev(values)


def _train_metrics_from_model(df, model, *, target, seed, test_size):
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataframe.")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, _, y_train, _ = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    preprocessor = build_preprocessor(df=df)
    X_train_processed = preprocessor.fit_transform(X_train)

    y_train_pred = model.predict(X_train_processed)

    return compute_model_metrics(y_true=y_train, y_pred=y_train_pred)


def _single_run(df, model_runner, model_name, *, seed):
    model_results = run_model(
        df=df,
        model_name=model_name,
        model_runner=lambda **kwargs: model_runner(
            **kwargs,
            seed=seed,
            test_size=TEST_SIZE,
            target="salario_classe"
        )
    )

    test_metrics = compute_model_metrics(
        y_true=model_results["results"]["y_test_true"],
        y_pred=model_results["results"]["y_test_pred"],
    )

    return model_results, test_metrics


def test_stability_across_seeds(df, model_runner, model_name):
    tpr_list = []
    fpr_list = []

    for seed in SEEDS:
        model_results, test_metrics = _single_run(
            df=df,
            model_runner=model_runner,
            model_name=model_name,
            seed=seed
        )

        tpr_list.append(test_metrics["tpr"])
        fpr_list.append(test_metrics["fpr"])

    tpr_mean, tpr_std = _mean_std(tpr_list)
    fpr_mean, fpr_std = _mean_std(fpr_list)

    status = "ok"
    if tpr_std > STABILITY_STD_THRESHOLD or fpr_std > STABILITY_STD_THRESHOLD:
        status = "suspect"

    return {
        "test": "stability",
        "tpr_mean": round(tpr_mean, 4),
        "tpr_std": round(tpr_std, 4),
        "fpr_mean": round(fpr_mean, 4),
        "fpr_std": round(fpr_std, 4),
        "status": status,
    }


def test_overfitting(df, model_runner, model_name):
    seed = SEEDS[0]

    model_results, test_metrics = _single_run(
        df=df,
        model_runner=model_runner,
        model_name=model_name,
        seed=seed
    )

    train_metrics = _train_metrics_from_model(
        df=df,
        model=model_results["results"]["model"],
        target="salario_classe",
        seed=seed,
        test_size=TEST_SIZE
    )

    tpr_gap = train_metrics["tpr"] - test_metrics["tpr"]

    status = "ok" if tpr_gap <= OVERFIT_GAP_THRESHOLD else "suspect"

    return {
        "test": "overfitting",
        "tpr_train": round(train_metrics["tpr"], 4),
        "tpr_test": round(test_metrics["tpr"], 4),
        "tpr_gap": round(tpr_gap, 4),
        "status": status,
    }


def test_underfitting(df, model_runner, model_name):
    seed = SEEDS[0]

    model_results, test_metrics = _single_run(
        df=df,
        model_runner=model_runner,
        model_name=model_name,
        seed=seed
    )

    train_metrics = _train_metrics_from_model(
        df=df,
        model=model_results["results"]["model"],
        target="salario_classe",
        seed=seed,
        test_size=TEST_SIZE
    )

    status = "ok"
    if (
        train_metrics["tpr"] < UNDERFIT_TPR_THRESHOLD
        and test_metrics["tpr"] < UNDERFIT_TPR_THRESHOLD
    ):
        status = "suspect"

    return {
        "test": "underfitting",
        "tpr_train": round(train_metrics["tpr"], 4),
        "tpr_test": round(test_metrics["tpr"], 4),
        "status": status,
    }


def test_target_distribution(df):
    if "salario_classe" not in df.columns:
        raise ValueError("Target 'salario_classe' not found in dataframe.")

    counts = df["salario_classe"].value_counts(normalize=True)
    minority_ratio = counts.min() if not counts.empty else 0.0

    status = "ok" if minority_ratio >= IMBALANCE_THRESHOLD else "suspect"

    return {
        "test": "target_distribution",
        "class_ratio": {int(k): round(v, 4) for k, v in counts.to_dict().items()},
        "minority_ratio": round(minority_ratio, 4),
        "status": status,
    }


def test_degenerate_model(df, model_runner, model_name):
    seed = SEEDS[0]

    model_results, _ = _single_run(
        df=df,
        model_runner=model_runner,
        model_name=model_name,
        seed=seed
    )

    y_pred = model_results["results"]["y_test_pred"]
    unique_predictions = len(set(y_pred))

    status = "ok" if unique_predictions > 1 else "suspect"

    return {
        "test": "degenerate_model",
        "unique_predictions": unique_predictions,
        "status": status,
    }


def run_sanity_checks(df_baseline, model_runner, model_name):
    results = []

    results.append(test_stability_across_seeds(
        df=df_baseline,
        model_runner=model_runner,
        model_name=model_name
    ))

    results.append(test_overfitting(
        df=df_baseline,
        model_runner=model_runner,
        model_name=model_name
    ))

    results.append(test_underfitting(
        df=df_baseline,
        model_runner=model_runner,
        model_name=model_name
    ))

    results.append(test_target_distribution(df_baseline))

    results.append(test_degenerate_model(
        df=df_baseline,
        model_runner=model_runner,
        model_name=model_name
    ))

    return results


if __name__ == "__main__":
    df_baseline, *_ = load_data()

    stats = compute_data_metrics(df_baseline["salario"])
    df_with_target = _add_salary_class(df_baseline, stats)

    results = run_sanity_checks(
        df_baseline=df_with_target,
        model_runner=run_logistic_regression,
        model_name="Logistic Regression"
    )

    print("\nSanity Check Results")
    print("=" * 40)

    for result in results:
        print("-" * 40)
        for key, value in result.items():
            print(f"{key}: {value}")
