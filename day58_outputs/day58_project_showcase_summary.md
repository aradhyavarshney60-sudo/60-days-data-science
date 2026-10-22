# Day 58 — Customer Intelligence Platform Showcase

## 1. Project Overview
The Customer Intelligence Platform is a data science capstone project focused on customer churn analysis, prediction, explainability, and business insights.

## 2. Business Problem
Customer churn occurs when customers stop using a service. This project explores churn patterns and demonstrates how predictive analytics can help teams investigate customer risk and plan retention activities.

## 3. Dataset
- Dataset: Telco Customer Churn
- Customer records: 7,043
- Source columns: 21
- Churn = Yes: 1,869
- Churn = No: 5,174

## 4. Analytics Workflow
1. Data preparation and cleaning
2. Model training and evaluation
3. Explainable AI using feature-importance and SHAP-based artifacts
4. Prediction API using FastAPI
5. Interactive demo using Streamlit
6. Business analysis through reports and visualizations

## 5. Model Evaluation
The documented Day 50 optimization results were:
- Gradient Boosting test accuracy: 80.62%
- Weighted F1 score: 79.79%

These are previously documented test-set metrics. They do not guarantee future predictions or prove production readiness or business impact.

## 6. Business Insights
The project includes churn-distribution analysis, contract and tenure visualizations, segment analysis, and business recommendations. These findings can help formulate retention experiments, which should be validated using real outcomes.

## 7. Live Demo Instructions

Start the API in the first terminal:

`python -m uvicorn day43_api:app --host 127.0.0.1 --port 8000`

Start Streamlit in a second terminal:

`python -m streamlit run day53.py`

Open `http://localhost:8501` and demonstrate a customer prediction, its displayed risk/probability, and the recommendation.

## 8. Reliability and Limitations
- Local API checks covered health, prediction, batch requests, and invalid-input validation.
- Local performance tests do not establish production-scale capacity.
- Model performance depends on data quality and evaluation methodology.
- Explanations show model behavior; they do not prove causation.
- Deployment, monitoring, and long-term model drift require further validation.

## 9. Future Improvements
- Strengthen deployment and security.
- Monitor input quality and API performance.
- Add model-drift checks.
- Gather user feedback.
- Measure actual retention outcomes before claiming business impact.

## 10. Repository
GitHub: https://github.com/aradhyavarshney60-sudo/60-days-data-science

**Day 58 deliverables:** Presentation deck and project showcase summary. Demo recording and final repository update remain to be completed.