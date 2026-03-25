import re
import matplotlib.pyplot as plt

def extract_epsilon(dataset_name):
    if dataset_name == "baseline":
        return 0
    match = re.search(r"eps_(\d+\.?\d*)", dataset_name)
    return float(match.group(1)) if match else None


def plot_tpr_evolution_combined(df):
    df = df.copy()
    df["epsilon"] = df["dataset"].apply(extract_epsilon)

    plt.figure()

    for group in df["grupo"].unique():
        subset = df[df["grupo"] == group].sort_values("epsilon")

        plt.plot(
            subset["epsilon"],
            subset["true positive rate"],
            marker="o",
            label=group
        )

    plt.title("Evolução do TPR por Grupo")
    plt.xlabel("Epsilon")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()