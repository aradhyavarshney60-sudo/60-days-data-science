# Day 53: Real-Time Customer Prediction Workflow

## Objective
Build an interactive Streamlit application connected to the existing FastAPI customer churn prediction service.

## Implementation
- Created a customer input form for demographic, service, contract, and billing details.
- Connected the form to the existing `/predict` API endpoint.
- Displayed the churn prediction, risk level, churn probability, and recommended action.
- Added a full API response viewer.
- Added an option to download the prediction result as a CSV file.
- Included API connection, timeout, and response error handling.

## Technology Stack
- Python
- Streamlit
- FastAPI
- Requests
- Pandas
- Existing trained customer churn model

## Demo Validation
The application was launched locally and a prediction was successfully returned.

Example displayed result:
- Churn prediction: No
- Risk level: Low
- Churn probability: 41.65%

These values are an example from one test submission, not an overall model performance metric.

## Workflow
1. User enters customer details.
2. Streamlit submits the details to the FastAPI prediction endpoint.
3. The API processes the input using the existing model.
4. The application displays the returned prediction and recommendation.
5. The user can inspect the API response or download the result.

## Limitations
Predictions depend on the existing model and the quality of submitted inputs. They are estimates, not guarantees. Further validation with representative customer cases is recommended.

## Conclusion
Day 53 demonstrates an interactive, API-connected customer churn prediction workflow, bringing machine learning predictions into a business-friendly interface.