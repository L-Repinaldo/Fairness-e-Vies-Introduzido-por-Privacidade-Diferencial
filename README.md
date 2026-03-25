# Fairness-e-Vies-Introduzido-por-Privacidade-Diferencial
---

# 📦 Projeto C — Fairness e Viés Introduzido por Privacidade Diferencial

## 🎯 Visão Geral

**Objetivo central:**
Avaliar se e como a aplicação de Privacidade Diferencial em dados tabulares de RH **introduz viés mensurável em decisões automatizadas**, mesmo quando a DP é corretamente aplicada.

**Pergunta de pesquisa:**

> A Privacidade Diferencial pode degradar fairness de modelos simples de forma assimétrica entre grupos?

**Hipóteses:**

* H1: O ruído da DP não afeta todos os grupos da mesma forma.
* H2: Grupos menores sofrem maior degradação de métricas de fairness sob ε baixo.
* H3: O trade-off privacidade × utilidade possui uma terceira dimensão: **fairness**.

---

## 🧩 Escopo Funcional

### ✅ O Projeto C faz

* Carrega datasets versionados do DP-Data-Pipeline
* Reutiliza a camada de input do Projeto B
* Treina **um classificador simples e fixo**
* Mede métricas de fairness por grupo
* Compara baseline vs versões DP (ε variados)
* Agrega métricas por seed e por tamanho de teste
* Gera tabelas e plots explicativos
* Produz síntese para discussão acadêmica

### ❌ O Projeto C não faz

* Não aplica DP
* Não faz tuning agressivo de modelos
* Não compara mecanismos de DP
* Não executa MIA
* Não busca SOTA
* Não altera datasets

---

## 🏗️ Arquitetura

Você pode literalmente clonar e podar:

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

---

## 🛠️ Ferramentas

**Stack:**

* Python
* Pandas / NumPy
* scikit-learn 
* Matplotlib 

**Classificador:**

* Regressão Logística 

Justificativa:
Modelo simples, interpretável, sensível a ruído.

---

## Como executar

1. Execute o pipeline principal:
```bash
python main.py
```

2. (Opcional) Rode o sanity check no baseline:
```bash
python sanity_check/sanity_model_check.py
```





## 📈 Resultados Esperados

O projeto deve produzir:

* Tabelas:

  * Variação das métricas utilizadas conforme ε
  
* Gráficos:

  * Evolução da taxa de verdadeiros positivos (tpr) por setor conforme ε

---

## 📜 Objetivos Acadêmicos

O Projeto C deve permitir afirmar:

* DP pode introduzir **viés mensurável**
* O impacto da DP **não é uniforme**
* Fairness deve ser considerada no trade-off junto com utilidade e segurança
* Privacidade não é neutra do ponto de vista distributivo
