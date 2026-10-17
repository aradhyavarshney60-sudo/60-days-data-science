
import requests
import streamlit as st
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Real-Time Customer Churn Prediction")
st.caption("Day 53 | Customer Intelligence Platform")
st.write(
    "Enter customer details below and submit the form "
    "to get a prediction from your existing machine learning API."
)

with st.sidebar:
    st.header("About")
    st.write("**Model:** Existing Day 43 churn model")
    st.write("**Prediction API:** FastAPI")
    st.write("**Interface:** Streamlit")
    st.info("Predictions are estimates, not guarantees.")

with st.form("customer_prediction_form"):
    st.subheader("1. Customer Profile")
    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])

    with c2:
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])

    with c3:
        monthly = st.number_input(
            "Monthly Charges", min_value=0.0,
            value=70.0, step=1.0
        )
        total = st.number_input(
            "Total Charges", min_value=0.0,
            value=840.0, step=10.0
        )

    st.subheader("2. Internet and Services")
    c1, c2, c3 = st.columns(3)

    with c1:
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        multiple = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        security = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"]
        )

    with c2:
        backup = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"]
        )
        device = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"]
        )
        tech = st.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"]
        )

    with c3:
        tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"]
        )

    st.subheader("3. Billing and Contract")
    c1, c2 = st.columns(2)

    with c1:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

    with c2:
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True,
    )

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    if total < 0 or monthly < 0:
        st.error("Charges cannot be negative.")
    else:
        try:
            with st.spinner("Getting prediction from the API..."):
                response = requests.post(
                    API_URL,
                    json=customer,
                    timeout=30,
                )

            if response.ok:
                result = response.json()
                st.subheader("Prediction Results")

                churn_value = result.get(
                    "churn_prediction",
                    result.get("prediction", "Not provided"),
                )

                if isinstance(churn_value, dict):
                    churn_value = churn_value.get(
                        "prediction",
                        churn_value.get("label", str(churn_value)),
                    )

                probability = result.get(
                    "churn_probability",
                    result.get("probability"),
                )

                risk = result.get("risk_level", "Not provided")
                recommendation = result.get(
                    "recommendation",
                    "No recommendation returned by the API.",
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Churn Prediction", str(churn_value))
                with col2:
                    st.metric(
                        "Risk Level",
                        str(risk),
                    )

                if isinstance(probability, (int, float)):
                    # API probabilities may be fractions or percentages.
                    probability_pct = (
                        probability * 100
                        if 0 <= probability <= 1
                        else probability
                    )
                    st.metric(
                        "Churn Probability",
                        f"{probability_pct:.2f}%",
                    )
                    st.progress(
                        min(max(probability_pct / 100, 0.0), 1.0)
                    )
                else:
                    st.info(
                        "The API did not return a separate probability field."
                    )

                st.markdown("### Recommended Action")
                if isinstance(recommendation, (dict, list)):
                    st.json(recommendation)
                else:
                    st.write(recommendation)

                with st.expander("View full API response"):
                    st.json(result)

                st.download_button(
                    "Download Prediction Result",
                    data=pd.DataFrame([{
                        "churn_prediction": str(churn_value),
                        "risk_level": str(risk),
                        "churn_probability": (
                            probability if probability is not None else ""
                        ),
                        "recommendation": str(recommendation),
                    }]).to_csv(index=False),
                    file_name="day53_prediction_result.csv",
                    mime="text/csv",
                )

            else:
                st.error(f"API error ({response.status_code})")
                st.code(response.text)

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to FastAPI. Start day43_api.py's "
                "server in another terminal, then submit again."
            )
        except requests.exceptions.Timeout:
            st.error("The API request timed out. Please try again.")
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
        except (ValueError, TypeError) as exc:
            st.error(f"Could not process the API response: {exc}")
