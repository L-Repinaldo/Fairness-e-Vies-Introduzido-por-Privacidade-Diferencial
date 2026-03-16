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




## 📈 Resultados Esperados

O projeto deve produzir:

* Tabelas:

  * Fairness por grupo × ε
* Gráficos:

  * Degradação de fairness conforme ε
* Síntese:

  * “DP introduziu assimetria?”
  * “Qual grupo foi mais afetado?”
  * “Fairness se degrada mais rápido que utilidade?”

---

## 📜 Objetivos Acadêmicos

O Projeto C deve permitir afirmar:

* DP pode introduzir **viés mensurável**
* O impacto da DP **não é uniforme**
* Fairness deve ser considerada no trade-off junto com utilidade e segurança
* Privacidade não é neutra do ponto de vista distributivo