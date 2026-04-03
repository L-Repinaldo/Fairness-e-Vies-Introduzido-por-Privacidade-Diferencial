# Fairness-e-Vies-Introduzido-por-Privacidade-Diferencial

## Visão Geral
Este projeto avalia se datasets tabulares de RH com privacidade diferencial (DP) podem induzir viés mensurável em modelos simples. A comparação é feita entre um baseline e versões DP (eps_0.1, eps_0.5, eps_1.0, eps_2.0) já disponíveis no repositório.

O pipeline executa um classificador fixo (Regressão Logística) e mede métricas globais e por grupo (setor), agregando resultados por seeds e tamanhos de teste.

## O que foi implementado
- Carregamento de datasets versionados em `data/datasets/<versao>`
- Criação do target `salario_classe` usando o limiar `mean + std` do salário
- Preprocessamento fixo com one-hot em `cargo` e `setor`
- Padronização de `idade`, `tempo_na_empresa`, `nota_media` quando presentes
- Treino com Regressão Logística (`liblinear`, `max_iter=1000`)
- Avaliação com matriz de confusão (tp, tn, fp, fn, tpr, fpr)
- Fairness por grupo **setor** com filtro de tamanho mínimo (>= 30)
- Agregação de métricas por seed e por test_size
- Visualização em tabelas e gráfico (TPR x epsilon por setor)

## O que não está no projeto
- Aplicação de DP (os datasets DP já estão prontos)
- Comparação de mecanismos DP
- Alteração dos datasets

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

## Dependências
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
Os CSVs são carregados de `data/datasets/<versao>` (definido em `config.py`) com os nomes:
- `baseline.csv`
- `dp_eps_0.1.csv`
- `dp_eps_0.5.csv`
- `dp_eps_1.0.csv`
- `dp_eps_2.0.csv`

Colunas esperadas:
- `salario` (usado para gerar `salario_classe`)
- `cargo` (obrigatória)
- `setor` (obrigatória)
- `idade` (opcional)
- `tempo_na_empresa` (opcional)
- `nota_media` (opcional)

## Saídas
- Tabelas com métricas globais e variação dos dados
- Tabela por setor com métricas de classificação
- Gráfico de evolução do TPR por setor em função do epsilon

As visualizações são exibidas via Matplotlib (não são salvas em disco por padrão).
