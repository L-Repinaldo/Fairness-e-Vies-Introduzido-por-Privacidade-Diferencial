from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def build_preprocessor(df):
    """
    Preprocessor fixo do experimento.

    Responsabilidades:
    - Selecionar features válidas presentes no dataset
    - Separar colunas categóricas e numéricas
    - Aplicar OneHotEncoder nas categóricas
    - Padronizar numéricas
    
    Parâmetros:
    - df: dataframe de entrada
    - groups: lista de colunas usadas para fairness (ex: ["cargo", "setor"])
    """

    groups = ["cargo", "setor"]

    categorical_candidates = ["cargo", "setor"]

    numerical_candidates = ["idade", "tempo_na_empresa", "nota_media"]

    categorical_cols = [
        col for col in categorical_candidates
        if col in df.columns
    ]

    numerical_cols = [
        col for col in numerical_candidates
        if col in df.columns
    ]

    if not categorical_cols and not numerical_cols:
        raise ValueError("Nenhuma feature válida encontrada para o experimento.")

    for group in groups:
        if group not in df.columns:
            raise ValueError(f"Coluna de grupo '{group}' não encontrada no dataset.")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    drop=None,                 
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_cols
            ),
            (
                "numerical",
                StandardScaler(),             
                numerical_cols
            ),
        ],
        remainder="drop"
    )

    return preprocessor