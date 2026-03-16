from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def run_logistic_regression( df, preprocessor, *, target= "salario", test_size=0.3, seed=42 ):
    """
    Executa Logistic Regression para o experimento de fairness.

    Responsabilidades:
    - Realizar split determinístico
    - Aplicar preprocessor apenas nas features
    - Preservar colunas de grupo para análise de fairness
    - Treinar modelo fixo
    - Retornar apenas dados necessários para avaliação
    """

    if target not in df.columns:
        raise ValueError(f"Target '{target}' não encontrado no dataset.")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=seed, stratify=y 
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = LogisticRegression(
        solver="liblinear", C=1.0,
        max_iter=1000, random_state=seed
    )

    model.fit(X_train_processed, y_train)

    y_test_pred = model.predict(X_test_processed)

    return {
        "y_test_true": y_test,
        "y_test_pred": y_test_pred,
        "model": model,
    }