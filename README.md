# Fairness-e-Vies-Introduzido-por-Privacidade-Diferencial

## Visao Geral
Este projeto avalia se datasets tabulares de RH com privacidade diferencial (DP) podem induzir vies mensuravel em modelos simples. A comparacao e feita entre um baseline e versoes DP (eps_0.1, eps_0.5, eps_1.0, eps_2.0) ja prontas no repositorio.

O pipeline executa um classificador fixo (Regressao Logistica) e mede metricas globais e por grupo (setor), agregando resultados por seeds e tamanhos de teste.

## O que foi implementado
- Carregamento de datasets versionados em `data/datasets/<versao>`
- Criacao do target `salario_classe` usando o limiar `mean + std` do salario
- Preprocessamento fixo com one-hot em `cargo` e `setor`
- Padronizacao de `idade`, `tempo_na_empresa`, `nota_media` quando presentes
- Treino com Regressao Logistica (`liblinear`, `max_iter=1000`)
- Avaliacao com matriz de confusao (tp, tn, fp, fn, tpr, fpr)
- Fairness por grupo **setor** com filtro de tamanho minimo (>= 30)
- Agregacao de metricas por seed e por test_size
- Visualizacao em tabelas e grafico (TPR x epsilon por setor)

## O que nao esta no projeto
- Aplicacao de DP (os datasets DP ja estao prontos)
- Ajuste de hiperparametros
- Comparacao de mecanismos DP
- Ataques de inferencia
- Alteracao dos datasets

## Estrutura
```text
project-c-fairness-dp/
├── data/
├── preprocessing/
├── model/
├── metrics/
├── plots/
├── experiments/
├── sanity_check/
├── config.py
└── main.py
```

## Dependencias
Instale os pacotes listados em `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Como executar
1. Pipeline principal:
```bash
python main.py
```

2. Sanity check (baseline):
```bash
python sanity_check/sanity_model_check.py
```

## Dados esperados
Os CSVs sao carregados de `data/datasets/<versao>` (definido em `config.py`) com os nomes:
- `baseline.csv`
- `dp_eps_0.1.csv`
- `dp_eps_0.5.csv`
- `dp_eps_1.0.csv`
- `dp_eps_2.0.csv`

Colunas esperadas:
- `salario` (usado para gerar `salario_classe`)
- `cargo` (obrigatoria)
- `setor` (obrigatoria)
- `idade` (opcional)
- `tempo_na_empresa` (opcional)
- `nota_media` (opcional)

## Saidas
- Tabelas com metricas globais e variacao dos dados
- Tabela por setor com metricas de classificacao
- Grafico de evolucao do TPR por setor em funcao do epsilon

As visualizacoes sao exibidas via Matplotlib (nao sao salvas em disco por padrao).
