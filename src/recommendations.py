from typing import Dict, List


def get_retention_recommendations(customer: Dict) -> List[str]:
    """
    Generate rule-based customer retention recommendations.
    """

    recommendations = []

    # Rule 1: Month-to-month contract
    if customer.get("Contract") == "Month-to-month":
        recommendations.append(
            "Offer a discount or incentive to move the customer "
            "to a one-year or two-year contract."
        )

    # Rule 2: Month-to-month + Fiber optic
    if (
        customer.get("Contract") == "Month-to-month"
        and customer.get("InternetService") == "Fiber optic"
    ):
        recommendations.append(
            "Prioritize this customer for proactive retention outreach "
            "because the month-to-month fiber optic segment has elevated churn risk."
        )

    # Rule 3: High monthly charges
    if customer.get("MonthlyCharges", 0) >= 70:
        recommendations.append(
            "Offer a personalized discount or service bundle "
            "to reduce the customer's monthly cost burden."
        )

    # Rule 4: New customer
    if customer.get("tenure", 0) < 12:
        recommendations.append(
            "Provide onboarding support and an early-loyalty incentive "
            "to strengthen customer engagement."
        )

    # Rule 5: No tech support
    if customer.get("TechSupport") == "No":
        recommendations.append(
            "Offer a free technical-support trial to increase service engagement."
        )

    # Rule 6: No online security
    if customer.get("OnlineSecurity") == "No":
        recommendations.append(
            "Offer a discounted online-security add-on."
        )

    # Rule 7: Electronic check
    if customer.get("PaymentMethod") == "Electronic check":
        recommendations.append(
            "Encourage migration to automatic payment methods "
            "for a more convenient billing experience."
        )

    # Fallback
    if not recommendations:
        recommendations.append(
            "Continue standard customer engagement and monitor churn risk."
        )

    return recommendations


def get_risk_level(churn_probability: float) -> str:
    """
    Convert churn probability into a risk category.
    """

    if churn_probability >= 0.70:
        return "High Risk"
    elif churn_probability >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


def build_customer_report(
    customer: Dict,
    churn_probability: float
) -> Dict:
    """
    Build a complete customer churn and retention report.
    """

    recommendations = get_retention_recommendations(customer)
    risk_level = get_risk_level(churn_probability)

    return {
        "churn_probability": round(churn_probability, 4),
        "risk_level": risk_level,
        "recommendations": recommendations
    }