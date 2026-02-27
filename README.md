# Fairness-e-Vi-s-Introduzido-por-Privacidade-Diferencial
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

## 🏗️ Arquitetura (reuso do Projeto B)

Você pode literalmente clonar e podar:

```text
project-c-fairness-dp/
├── data/              # reuso do loader e estrutura
├── preprocessing/    # mesmo padrão do Projeto B
├── model/            # classificador simples
├── metrics/          # fairness metrics
├── analysis/         # análise por ε e por grupo
├── plots/            # visualizações
├── experiments/      # definição de cenários
├── config.py         # DATASET_VERSION, EPS_LIST, TARGET, GROUPS
└── main.py
```

---

## 🛠️ Ferramentas

**Stack (mesma filosofia do Projeto B):**

* Python
* Pandas / NumPy
* scikit-learn (classificador simples)
* Matplotlib / Seaborn (plots)

**Classificador (fixo, sem tuning):**

* Regressão Logística **ou**
* Árvore de decisão rasa

Justificativa:
Modelo simples, interpretável, sensível a ruído.

---

## 📊 Métricas de Fairness (mínimo necessário)

Você não precisa inventar nada:

Por grupo (ex: setor, cargo, faixa salarial):

* Acurácia por grupo
* FPR por grupo
* FNR por grupo
* Taxa de predição positiva por grupo

Métricas comparativas:

* Δ Acurácia entre grupos
* Δ FPR / Δ FNR
* Variação dessas diferenças entre baseline e DP

---

## 🔎 Variáveis do Experimento

No `config.py`:

```text
DATASET_VERSION = "v-YYYY-MM-DD_HH-MM-SS"
TARGET = "coluna_alvo"
GROUPS = ["setor", "cargo"]   # ou outra categórica
EPS_LIST = ["baseline", "dp_eps_0.1", "dp_eps_1.0"]
```

Isso mantém o padrão de reprodutibilidade do Projeto B.

---

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

---

## ⏱️ Estimativa de Tempo (realista)

Assumindo que você vai reaproveitar código do Projeto B:

| Etapa                              | Tempo   |
| ---------------------------------- | ------- |
| Definir alvo + grupos              | 0,5 dia |
| Adaptar loader / config            | 0,5 dia |
| Implementar métricas de fairness   | 1 dia   |
| Criar pipeline de experimento      | 1 dia   |
| Gerar plots e tabelas              | 0,5 dia |
| Escrever README + estrutura artigo | 1 dia   |

👉 Total: **~4 a 5 dias úteis** para fechar o projeto técnico.

---

## 🧠 Diferencial Científico do Projeto C

Você passa a ter:

* Artigo 1:

  > “DP × Utilidade × Vazamento (ML + MIA)”

* Artigo 2:

  > “DP × Fairness × Impacto Social em decisões automatizadas”

Isso te coloca num patamar de:

> não só medir performance,
> mas discutir consequências.

---
* o README do Projeto C
* e o esqueleto do artigo (introdução, problema, método, resultados, discussão).
