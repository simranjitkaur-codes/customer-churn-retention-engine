# Week 1 Business Insights

## 1. Scope

This report summarizes the Week 1 exploratory analysis of customer churn.

The analysis uses the training dataset only to avoid introducing information from the held-out test set.

- Training customers: 5,616
- Overall churn rate: 26.44%
- Analysis areas:
  - Contract type
  - Payment method
  - Internet service
  - Online security
  - Tech support
  - Paperless billing
  - Tenure
  - Monthly charges
  - Contract × Internet Service interaction

---

## 2. Overall Churn

Out of 5,616 training customers:

- 4,131 customers stayed
- 1,485 customers churned
- Overall churn rate: **26.44%**

This establishes the baseline against which individual customer segments can be compared.

---

## 3. Highest-Risk Customer Segments

Several customer segments show substantially higher churn than the overall 26.44% baseline.

| Risk Segment | Customers | Churned | Churn Rate |
|---|---:|---:|---:|
| Electronic check | 1,905 | 861 | **45.20%** |
| Month-to-month contract | 3,078 | 1,313 | **42.66%** |
| Fiber optic | 2,469 | 1,045 | **42.32%** |
| No online security | 2,793 | 1,165 | **41.71%** |
| No tech support | 2,780 | 1,153 | **41.47%** |
| Paperless billing | 3,365 | 1,125 | **33.43%** |

These segments have churn rates considerably above the overall churn rate.

The electronic check segment has the highest churn rate among the selected risk segments at **45.20%**.

---

## 4. Contract Type

Contract type shows one of the strongest differences in churn.

- Month-to-month: **42.66%**
- One year: **11.17%**
- Two year: **2.95%**

Customers on month-to-month contracts have substantially higher churn than customers on longer-term contracts.

### Business implication

Contract duration may be an important indicator of customer retention.

A potential business hypothesis is that customers with flexible month-to-month contracts may have lower switching barriers and therefore may be easier to lose.

---

## 5. Contract × Internet Service Interaction

The interaction analysis shows that churn varies substantially across combinations of contract type and internet service.

| Contract | DSL | Fiber optic | No Internet |
|---|---:|---:|---:|
| Month-to-month | 32.0% | **54.7%** | 17.5% |
| One year | 8.8% | 19.9% | 2.3% |
| Two year | 1.6% | 8.3% | 0.8% |

The highest observed churn rate is among:

**Month-to-month + Fiber optic: 54.7%**

This is substantially higher than the overall churn rate of 26.44%.

### Business implication

The results suggest that contract type and internet service may interact in their relationship with churn.

This combination could therefore be useful as a feature or interaction term in later predictive modeling.

However, this analysis is observational and does not establish that either factor directly causes churn.

---

## 6. Payment Method

Payment method also shows meaningful differences in churn.

The electronic check segment has a churn rate of **45.20%**, making it the highest-risk segment among the selected Day 5 risk indicators.

### Business hypothesis

Customers using electronic checks may have different payment behavior or customer characteristics associated with higher churn.

Further analysis is required before concluding that changing payment method itself would reduce churn.

---

## 7. Online Security and Technical Support

Customers without online security have a churn rate of **41.71%**.

Customers without tech support have a churn rate of **41.47%**.

Both are substantially higher than the overall churn rate of **26.44%**.

### Business hypothesis

Lower adoption of support and security services may be associated with customers who are less engaged with the service ecosystem or who have different service needs.

These features should therefore be considered during predictive modeling.

---

## 8. Paperless Billing

Customers with paperless billing show a churn rate of **33.43%**, compared with **15.99%** for customers without paperless billing.

The difference is substantial.

### Business hypothesis

Paperless billing may be acting as an indicator of customer characteristics or digital-service usage rather than directly causing churn.

Further investigation is required before treating paperless billing as a causal churn driver.

---

## 9. Tenure and Monthly Charges

The Day 4 analysis identified additional patterns.

### Tenure

- Churned customers:
  - Mean tenure: **18.35 months**
  - Median tenure: **10 months**
- Stayed customers:
  - Mean tenure: **37.50 months**
  - Median tenure: **37 months**

Churned customers generally have much shorter tenure than customers who stayed.

### Monthly Charges

- Churned customers:
  - Mean monthly charges: **75.13**
  - Median monthly charges: **79.90**
- Stayed customers:
  - Mean monthly charges: **61.22**
  - Median monthly charges: **64.50**

Churned customers also have higher average and median monthly charges.

### Business implication

Early-tenure customers and customers with higher monthly charges may deserve additional attention in retention strategies.

---

## 10. Key Business Hypotheses

Based on Week 1 EDA, the following hypotheses should be investigated further:

1. Month-to-month customers may have a higher likelihood of churn because of lower contractual commitment.

2. The combination of month-to-month contracts and fiber-optic service may represent a particularly high-risk customer segment.

3. Electronic-check customers may have characteristics associated with increased churn.

4. Customers without online security or technical support may have higher churn risk.

5. Short-tenure customers may be more vulnerable to churn.

6. Higher monthly charges may be associated with increased churn.

7. Paperless billing may be associated with customer characteristics that correlate with churn rather than directly causing churn.

These are hypotheses, not causal conclusions.

---

## 11. Modeling Implications

The Week 1 analysis suggests several features that should be considered during predictive modeling:

- Contract
- PaymentMethod
- InternetService
- OnlineSecurity
- TechSupport
- PaperlessBilling
- Tenure
- MonthlyCharges

Potential interaction features should also be considered, particularly:

- Contract × InternetService

The large difference between churn rates across several categorical segments suggests that these variables may provide useful predictive information.

---

## 12. Week 1 Conclusion

The exploratory analysis identifies a clear group of higher-risk customer segments.

The strongest individual risk indicators examined in Day 5 include:

- Electronic check: **45.20%**
- Month-to-month contract: **42.66%**
- Fiber optic: **42.32%**
- No online security: **41.71%**
- No tech support: **41.47%**

The strongest interaction identified is:

**Month-to-month + Fiber optic: 54.7% churn**

Combined with the Day 4 findings around shorter tenure and higher monthly charges, these results provide a strong foundation for the next phase of the project: feature engineering and predictive churn modeling.

All findings are based on training data only. The held-out test set remains untouched.
