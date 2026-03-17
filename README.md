# Financial_Health_Index_Prediction_Challenge
https://zindi.africa/competitions/dataorg-financial-health-prediction-challenge

## INTRODUCTION
The small and medium enterprise (SME) sector in Southern Africa represents the most critical frontier for economic development, employment creation, and social innovation. Across Eswatini, Lesotho, Zimbabwe, and Malawi, these businesses operate as the primary livelihood for millions, yet they remain fundamentally fragile and largely excluded from the formal financial systems that might otherwise provide stability during times of crisis. Traditional economic metrics, such as annual profit or gross revenue, are increasingly recognized as insufficient for capturing the actual well-being of an SME in a developing economy. Instead, a more nuanced understanding is required—one that considers the interdependencies between savings habits, debt management, resilience to idiosyncratic shocks, and the qualitative nature of financial inclusion. The Financial Health Index (FHI) addresses this need by providing a composite measure that classifies enterprises into Low, Medium, or High financial health. However, the construction of reliable machine learning models to predict the FHI is hampered by the nature of the primary data source: survey responses that are frequently incomplete, inconsistent, and influenced by a complex array of behavioral and cultural factors.

## FINANCIAL HEALTH INDEX (FHI) DEFINED 
The FHI serves as a redefined metric for SME wellbeing, moving the focus from simple profitability to a holistic view of resilience and opportunity. The index is built across 4 dimensions. These dimensions are not isolated; they interact in ways that provide hidden information for data recovery.
| Dimension | Definition | Dataset proxies | 
| :--- | :--- | :--- | 
| **Savings & Assets** | The accumulation of liquid and fixed capital to support business continuity. | personal_income, has_credit_card, has_debit_card, uses_friends_family_savings  | 
| **Debt & Repayment** | The ability to service liabilities without compromising operational integrity. | has_loan_account, offers_credit_to_customers, business_expenses  | 
| **Resilience to shocks** | Buffers against unexpected events like climate events or health crises. | has_insurance, medical_insurance, attitude_worried_shutdown  | 
| **Access to credit** | Integration into formal and informal financial service networks. | has_mobile_money, has_internet_banking, problem_sourcing_money  | 

## CHALLENGE OBJECTIVE
Develop a robust machine learning model to accurately predict the Financial Health Index (FHI) class—Low, Medium, or High—of small and medium-sized enterprises (SMEs) using socio-economic, demographic, and business data from Eswatini, Lesotho, Zimbabwe, and Malawi. 

## REPOSITORY LAYOUT
## CODING ENVIRONMENT
## HOW TO RUN THE CODE
## NOTEBOOK RUNTIME
## ARCHITECTURAL DIAGRAM
## DATA DESCRIPTION
The dataset comprises real-world survey responses collected from small and medium-sized enterprise (SME) owners across four Southern African nations: Eswatini, Lesotho, Malawi, and Zimbabwe.
As this is raw survey data, it authentically reflects the messy, subjective nature of on-the-ground data collection. The data contained missing values, "Don't know" responses, refusals to answer, and extreme outliers (e.g., currency inflation or data-entry typos). Successfully navigating these anomalies was a core part of the challenge.

### Files
1. Train.csv : The training set, containing the survey features and the target variable.
2. Test.csv : The test set. You must predict the financial health class for these unseen businesses.

## Target Variable
Target: The Financial Health Index (FHI) classification of the business.
  - Low (Highly vulnerable, informal, or distressed)
  - Medium (Stable but potentially lacking formal infrastructure)
  - High (Resilient, formalized, and financially integrated)

<img src="assets/overall_class_imbalance.png" width="400"><img src="assets/country_class_imbalance.png" width="400">
