# Sanity Check

## Purpose
This folder validates model behavior on the **baseline dataset** before running any differential privacy experiments.

## What is being tested
- Model stability
- Signs of overfitting or underfitting
- Consistency of metrics

## Important
- This is **NOT** part of the main pipeline
- It **does NOT** modify datasets
- It is intended only for diagnostic validation

## How to run
From the project root:

```bash
python sanity_check/sanity_model_check.py
```
