# Project knowledge base (example)

Copy this file to `knowledge/projects.md` (gitignored) and replace the blocks with your own
projects. One `## <Project name>` block per project. `stack`, `metrics` and `keywords` are
comma-separated lists; use `-` when a field is empty. Agents may only use facts written here
(plus the CV) when rewriting, so every number must be true.

## Support Ticket Triage Assistant
category: work
employer: Example Analytics Ltd
period: Mar 2023 to Present
problem: Support agents spent hours a day classifying and routing inbound tickets by hand, with inconsistent priority labels.
built: A fine-tuned small language model that classifies tickets by product area and urgency, served through a FastAPI endpoint with a human review queue for low-confidence cases. Retrained monthly from reviewer corrections.
stack: Python, FastAPI, Hugging Face Transformers, LoRA, PostgreSQL, Docker, GitHub Actions
metrics: 85% of tickets routed without human correction, 40k tickets/month, triage time cut from 6 min to 45 s
keywords: text classification, fine-tuning, human-in-the-loop, MLOps

## Bike-Share Demand Forecaster
category: personal
employer: -
period: Sep 2021 to Feb 2022
problem: Predict hourly bike-share demand per station so operators can rebalance bikes ahead of peaks.
built: Gradient-boosted regression on weather, calendar and station history features, compared against a seasonal baseline; results served in a small Streamlit dashboard.
stack: Python, Pandas, scikit-learn, XGBoost, Streamlit
metrics: -
keywords: time series, regression, feature engineering, dashboard
