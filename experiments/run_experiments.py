from statistics import mean

from metrics import compute_data_metrics

from . import run_model, model_metrics

SEEDS = [42, 123, 2026]
TEST_SIZES = [0.2, 0.3]

def _aggregate_metrics(metrics_list):
    return {
        k: round(mean(m[k] for m in metrics_list), 3)
        for k in metrics_list[0].keys()
    }


def run_experimemnt(model_name, model_runner, datasets, dataset_names):

    """
    Protocolo experimental padrão do projeto.

    Este método:
    - Chama os métodos responsáveis pelo experimento da aplicação 
    - organiza os resultados

    NÃO:
    - altera datasets
    - Executa experimento
    """

    data_stats = {}
    model_experiment_output = {}


    for name, df in zip(dataset_names, datasets):

        data_stats[name] = compute_data_metrics(df['salario'])

        model_runs = []

        for seed in SEEDS:
            for test_size in TEST_SIZES:

                model_results = run_model(df=df, model_name=model_name,
                                            model_runner=lambda **kwargs: model_runner(
                                                **kwargs,
                                                seed= seed,
                                                test_size= test_size
                                            )
                                        )
                model_metrics_values = model_metrics(model_output= model_results, dataset_name=name , model_name=model_name) 
                model_runs.append(model_metrics_values['results'])
        
        model_experiment_output[name] = {
            "model_name": model_name,
            "results": _aggregate_metrics(model_runs)
        }
            
    return data_stats, model_experiment_output