from preprocessing import build_preprocessor
from metrics import compute_model_metrics

def run_model( df, model_runner, model_name):
    """
    Protocolo experimental padrão do projeto.

    Este método:
    - executa o modelo como instrumento de medição
    - mede utilidade
    - mede estabilidade
    - executa ataque de inferência
    - organiza os resultados

    NÃO:
    - altera datasets
    - aplica DP
    - otimiza modelos
    """


    preprocessor = build_preprocessor(df=df)

    run_output = model_runner(
        df=df,
        preprocessor=preprocessor,
    )

    return {
        "model_name": model_name,
        "results": run_output,
    }

def model_metrics(model_output, dataset_name, model_name):

    output = compute_model_metrics(
        y_true=model_output['results']["y_test_true"],
        y_pred=model_output['results']["y_test_pred"],
    )


    return {
        "model_name": model_name,
        "results": output,
    }