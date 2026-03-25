import matplotlib.pyplot as plt

def plot_table(df, title="Tabela de Resultados"):
    fig, ax = plt.subplots()

    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    plt.title(title)

    plt.tight_layout()
    plt.show()