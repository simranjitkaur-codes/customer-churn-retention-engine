from src.recommendations import (
    get_retention_recommendations,
    get_risk_level,
    build_customer_report
)


def test_high_risk():
    assert get_risk_level(0.70) == "High Risk"
    assert get_risk_level(0.90) == "High Risk"


def test_medium_risk():
    assert get_risk_level(0.40) == "Medium Risk"
    assert get_risk_level(0.69) == "Medium Risk"


def test_low_risk():
    assert get_risk_level(0.39) == "Low Risk"
    assert get_risk_level(0.10) == "Low Risk"


def test_month_to_month_recommendation():
    customer = {
        "Contract": "Month-to-month"
    }

    recommendations = get_retention_recommendations(customer)

    assert len(recommendations) >= 1
    assert any("contract" in r.lower() for r in recommendations)


def test_high_risk_combination():
    customer = {
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic"
    }

    recommendations = get_retention_recommendations(customer)

    assert len(recommendations) >= 2


def test_stable_customer_fallback():
    customer = {
        "Contract": "Two year",
        "InternetService": "No",
        "MonthlyCharges": 30,
        "tenure": 50,
        "TechSupport": "Yes",
        "OnlineSecurity": "Yes",
        "PaymentMethod": "Bank transfer"
    }

    recommendations = get_retention_recommendations(customer)

    assert len(recommendations) == 1
    assert "monitor" in recommendations[0].lower()


def test_missing_keys():
    customer = {}

    recommendations = get_retention_recommendations(customer)

    assert isinstance(recommendations, list)
    assert len(recommendations) >= 1


def test_recommendations_are_strings():
    customer = {
        "Contract": "Month-to-month",
        "MonthlyCharges": 80
    }

    recommendations = get_retention_recommendations(customer)

    assert all(isinstance(r, str) for r in recommendations)


def test_customer_report_structure():
    customer = {
        "Contract": "Month-to-month"
    }

    report = build_customer_report(customer, 0.75)

    assert "churn_probability" in report
    assert "risk_level" in report
    assert "recommendations" in report


def test_probability_rounding():
    customer = {}

    report = build_customer_report(
        customer,
        0.823456
    )

    assert report["churn_probability"] == 0.8235