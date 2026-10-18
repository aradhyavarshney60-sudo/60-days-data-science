
import json
import statistics
import time
from datetime import datetime, timezone

import requests

API_URL = "http://127.0.0.1:8000"
OUTPUT_FILE = "day55_performance_report.md"

SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
}


def timed_request(method, url, **kwargs):
    start = time.perf_counter()
    response = requests.request(
        method,
        url,
        timeout=10,
        **kwargs,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return response, elapsed_ms


def main():
    print("=" * 55)
    print("DAY 55 - API PERFORMANCE AND RELIABILITY TEST")
    print("=" * 55)

    results = []

    # 1. Health endpoint
    try:
        response, elapsed = timed_request(
            "GET", f"{API_URL}/health"
        )
        response.raise_for_status()
        health = response.json()

        print(f"\nHealth check: PASS ({elapsed:.2f} ms)")
        print(json.dumps(health, indent=2))
        results.append(("Health check", "PASS", elapsed))

    except requests.RequestException as error:
        print(f"\nAPI unavailable: {error}")
        print(
            "Start the API in another terminal using:\n"
            "python -m uvicorn day43_api:app --host "
            "127.0.0.1 --port 8000"
        )
        return

    # 2. Single prediction latency
    single_times = []

    for attempt in range(10):
        try:
            response, elapsed = timed_request(
                "POST",
                f"{API_URL}/predict",
                json=SAMPLE_CUSTOMER,
            )
            response.raise_for_status()
            payload = response.json()
            single_times.append(elapsed)

            if attempt == 0:
                print("\nSample prediction:")
                print(json.dumps(payload, indent=2))

        except requests.RequestException as error:
            print(f"Single prediction failed: {error}")
            results.append(
                ("Single prediction", "FAIL", None)
            )
            break

    if single_times:
        results.append(
            ("Single prediction", "PASS", statistics.mean(single_times))
        )

    # 3. Batch prediction latency
    batch_times = []

    for batch_size in (1, 10, 50):
        customers = [
            SAMPLE_CUSTOMER.copy()
            for _ in range(batch_size)
        ]

        try:
            response, elapsed = timed_request(
                "POST",
                f"{API_URL}/predict-batch",
                json=customers,
            )
            response.raise_for_status()
            payload = response.json()

            returned = len(payload.get("predictions", []))
            passed = returned == batch_size

            print(
                f"Batch size {batch_size}: "
                f"{'PASS' if passed else 'FAIL'} "
                f"({elapsed:.2f} ms; {returned} results)"
            )

            results.append(
                (
                    f"Batch size {batch_size}",
                    "PASS" if passed else "FAIL",
                    elapsed,
                )
            )
            batch_times.append((batch_size, elapsed))

        except requests.RequestException as error:
            print(f"Batch size {batch_size} failed: {error}")
            results.append(
                (f"Batch size {batch_size}", "FAIL", None)
            )

    # 4. Invalid input validation
    try:
        response, elapsed = timed_request(
            "POST",
            f"{API_URL}/predict",
            json={"SeniorCitizen": 5},
        )

        validation_passed = response.status_code == 422
        print(
            f"\nInvalid input validation: "
            f"{'PASS' if validation_passed else 'CHECK'} "
            f"(HTTP {response.status_code})"
        )
        results.append(
            (
                "Invalid input validation",
                "PASS" if validation_passed else "CHECK",
                elapsed,
            )
        )

    except requests.RequestException as error:
        print(f"Validation test failed: {error}")
        results.append(
            ("Invalid input validation", "FAIL", None)
        )

    # 5. Write report
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("# Day 55: API Performance and Reliability Report\n\n")
        file.write(
            f"- Test date (UTC): "
            f"{datetime.now(timezone.utc).isoformat()}\n"
        )
        file.write(f"- API URL: `{API_URL}`\n\n")
        file.write("## Test Results\n\n")
        file.write("| Test | Status | Latency (ms) |\n")
        file.write("|---|---|---:|\n")

        for name, status, latency in results:
            latency_text = (
                f"{latency:.2f}"
                if latency is not None
                else "N/A"
            )
            file.write(
                f"| {name} | {status} | {latency_text} |\n"
            )

        if single_times:
            file.write("\n## Single Prediction Latency\n\n")
            file.write(
                f"- Requests completed: {len(single_times)}\n"
            )
            file.write(
                f"- Average: {statistics.mean(single_times):.2f} ms\n"
            )
            file.write(
                f"- Median: {statistics.median(single_times):.2f} ms\n"
            )
            file.write(
                f"- Maximum: {max(single_times):.2f} ms\n"
            )

        file.write(
            "\n## Limitations\n\n"
            "Results reflect a small local test, not a production "
            "load test. Latency depends on hardware, server state, "
            "network conditions, and warm-up effects. No performance "
            "improvement should be claimed without comparing against "
            "a documented baseline.\n"
        )

    print(f"\nReport saved to: {OUTPUT_FILE}")
    print("Performance and reliability tests completed.")


if __name__ == "__main__":
    main()
