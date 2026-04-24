# Fairness e Viés Induzido por Privacidade Diferencial

## Visão Geral

Este repositório investiga se a aplicação de Privacidade Diferencial (DP) em datasets tabulares de RH pode introduzir **viés mensurável** em modelos de Machine Learning simples.

A análise é conduzida comparando:

- Um dataset **baseline (sem DP)**  
- Múltiplas versões **privatizadas** com diferentes níveis de ε:
  - `0.1`, `0.5`, `1.0`, `2.0`

O objetivo não é otimizar modelos, mas **observar como o ruído afeta o comportamento estatístico e a equidade do aprendizado**.

---

## Papel no Ecossistema do Projeto

Este repositório atua como **módulo de análise de fairness**, dentro de um pipeline maior:

- Dados já foram:
  - gerados (sistema de RH)
  - privatizados (pipeline de DP)
- Este projeto:
  - **não altera dados**
  - **não aplica DP**
  - apenas mede efeitos

Aqui, o foco não é utilidade global ou ataque, mas:

> **como a DP impacta grupos de forma desigual**

---

## Objetivo

Avaliar empiricamente se diferentes níveis de ε:

- Alteram a distribuição dos dados  
- Afetam métricas de classificação  
- Introduzem ou amplificam viés entre grupos (setor)  
- Criam distorções sistemáticas no TPR/FPR  

A pergunta central é:

> **Privacidade Diferencial pode induzir unfairness mesmo em modelos simples?**

---

## Abordagem Experimental

O pipeline segue uma estrutura fixa para isolar o efeito da DP.

### Modelo

- Regressão Logística (`liblinear`)
- `max_iter = 1000`
- Sem tuning

Justificativa:

- Modelo simples e interpretável  
- Sensível a mudanças na distribuição  
- Evita mascarar efeitos do ruído  

---

### Target

  - `salario_classe`
- Definido como:

``` python
    salario > mean + std
```

## Preprocessamento
  - One-hot encoding:
    - cargo
    - setor
  - Padronização (quando presentes):
    - idade
    - tempo_na_empresa
    - nota_media

Pipeline fixo para garantir comparabilidade.

---

## Avaliação
  ***Métricas Globais***
    - TP, TN, FP, FN
    - TPR (recall positivo)
    - FPR
  ***Fairness por Grupo (setor)***
    - Métricas calculadas por setor
    - Filtro: grupos com ≥ 30 amostras

---

## Controle Experimental
  - Múltiplas seeds
  - Diferentes test_size
  - Agregação dos resultados
Isso reduz variância e evita conclusões baseadas em flutuação aleatória.

---

## Escopo do Projeto

Este repositório:

  - Carrega datasets versionados
  - Executa pipeline fixo de ML
  - Mede métricas globais e por grupo
  - Compara comportamento entre níveis de ε
  - Gera tabelas e visualizações

Este repositório ***não***:

  - Aplica Privacidade Diferencial
  - Gera ou modifica dados
  - Compara mecanismos de DP
  - Otimiza modelos
    
---

## Estrutura do Projeto

    project-fairness-dp/
    ├── data/
    ├── preprocessing/
    ├── model/
    ├── metrics/
    ├── plots/
    ├── experiments/
    ├── sanity_check/
    ├── config.py
    └── main.py
  
---

## Dados
 Datasets esperados em:
 
         data/datasets/<versao>/

Arquivos:

    baseline.csv
    dp_eps_0.1.csv
    dp_eps_0.5.csv
    dp_eps_1.0.csv
    dp_eps_2.0.csv

Colunas esperadas

 - Obrigatórias:

     - salario
     - cargo
     - setor

- Opcionais:

    - idade
    - tempo_na_empresa
    - nota_media
 
---

## Execução

Pipeline principal

    python main.py

Sanity check (baseline)

    python sanity_check/sanity_model_check.py
    
---

## Saídas

O projeto gera:

### Tabelas

  - Métricas globais
  - Variação estatística dos dados
  - Resultados por setor

### Visualizações

  - Evolução do TPR por setor em função de ε

As visualizações são exibidas via Matplotlib (não persistidas por padrão).

---

## Interpretação dos Resultados
Os resultados permitem observar:

  - Se a DP degrada desempenho de forma uniforme
  - Se certos setores são mais afetados que outros
  - Se há distorções sistemáticas no TPR/FPR
  - Se o ruído cria assimetria entre grupos
  - 
***Importante:***

        Queda de performance ≠ viés
        Viés surge quando o impacto não é uniforme entre grupos
    
---

## Reprodutibilidade

Os experimentos são determinísticos dado:

  - Dataset versionado
  - Seeds fixas
  - Pipeline imutável

A versão ativa é definida em:

    config.py

---

## Limitações

  - Apenas um modelo (Regressão Logística)
  - Apenas um tipo de feature engineering
  - Apenas fairness por setor
  - Não mede causalidade, apenas correlação
  
---

##Imagens 

***Variação do Dados***
<img width="640" height="480" alt="data" src="https://github.com/user-attachments/assets/4ee032a8-61a2-4baa-b32d-d6eef440cc3d" />

***Resultados Logistic Regression***
<img width="1366" height="655" alt="Figure_1" src="https://github.com/user-attachments/assets/104dd66d-0042-4984-bdc1-333f8a834acc" />

***Evolução do TPR por setor***
<img width="1366" height="655" alt="setor tpr" src="https://github.com/user-attachments/assets/707ac1c0-2cdc-4e5b-9829-de5a76e492d6" />

***Resultados por Setor***
<img width="1366" height="655" alt="Resultados setor" src="https://github.com/user-attachments/assets/7952dc62-d33c-418a-a9ef-6b3d7b42288f" />

---

## Motivação

Este projeto foi desenhado para:
  - Isolar o efeito da Privacidade Diferencial
  - Analisar impactos além da utilidade média
  - Tornar visível o comportamento por grupo
  - Explorar a relação entre ruído e fairness
    
---

## Observações

  - Dados são simulados
  - Uso acadêmico
  - Resultados têm caráter explicativo, não prescritivo

---

## Licença
Uso educacional e acadêmico.
