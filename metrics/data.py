def data_metrics(series):

    mean = series.mean()
    median = series.median()
    std = series.std()

    return {
        "mean": mean,
        "median": median,
        "std": std
    }