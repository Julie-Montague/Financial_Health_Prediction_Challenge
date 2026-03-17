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

### Target Variable
Target: The Financial Health Index (FHI) classification of the business.
  - Low (Highly vulnerable, informal, or distressed)
  - Medium (Stable but potentially lacking formal infrastructure)
  - High (Resilient, formalized, and financially integrated)
    
<p align="center">
  <table>
    <tr>
      <td align="center"><img src="assets/overall_class_imbalance.png" width="400"></td>
      <td align="center"><img src="assets/country_class_imbalance.png" width="400"></td>
    </tr>
    <tr>
      <td align="center"><b>Overall Imbalance</b></td>
      <td align="center"><b>Imbalance by Country</b></td>
    </tr>
  </table>
</p>

As seen in the overall distribution, the majority of businesses fall into the **Low** FHI category. These are typically informal, subsistence-level micro-businesses operating entirely in cash. On the opposite end of the spectrum, the **High** class is extremely rare, representing businesses that have successfully scaled and integrated into the formal banking sector. 
This structural imbalance persists across all four nations (Eswatini, Lesotho, Malawi, and Zimbabwe). From a machine learning perspective, this causes standard algorithms to become overly conservative, naturally defaulting to "Low" or "Medium" predictions to minimize loss. 
To counteract this and successfully capture the rare "High" performing SMEs, this project implements advanced techniques including:

1. **SMOTE Oversampling:** Synthetically generating realistic data points for "High" businesses during cross-validation to teach the model their specific traits.
2. **Confident Learning (Cleanlab):** Dropping highly-disputed, noisy majority-class survey responses rather than allowing them to dilute the tree's learning process.

### Data Dictionary
| Category | Feature Name | Description |
|---|---|---|
| Demographics | country | The country of operation (Eswatini, Lesotho, Malawi, Zimbabwe). |
| Demographics | owner_age | The age of the primary business owner. |
| Demographics | owner_sex | The gender of the primary business owner. |
| Firmographics | business_age | The age of the business in years. |
| Firmographics | keeps_financial_records | Whether the business tracks finances (for example: Yes, No, Sometimes, Missing). |
| Financial/Cash | business_turnover | Total raw revenue generated by the business. |
| Financial/Cash | business_expenses | Total operating costs of the business. |
| Financial/Cash | personal_income | The amount of business income withdrawn by the owner for personal use. |
| Financial/Cash | current_problem_cash_flow | Whether the business is currently experiencing cash flow shortages. |
| Financial/Cash | problem_sourcing_money | The level of difficulty in obtaining capital for operations. |
| Digital Access | has_cellphone | Indicates ownership of a basic cellphone. |
| Digital Access | has_mobile_money | Indicates usage of mobile money services (for example, M-Pesa). |
| Digital Access | has_internet_banking | Indicates usage of a formal web-based banking portal. |
| Formal Credit | has_debit_card | Ownership of a bank debit card. |
| Formal Credit | has_credit_card | Ownership of a bank credit card. |
| Formal Credit | has_loan_account | Indicates an active formal loan with a registered financial institution. |
| Informal Funding | uses_friends_family_savings | Reliance on personal networks to fund the business. |
| Informal Funding | uses_informal_lender | Reliance on informal or unregistered money lenders, often at high interest. |
| Compliance | compliance_income_tax | Whether the business pays formal national income tax. |
| Compliance | offers_credit_to_customers | Whether the business extends buy-now-pay-later credit to customers. |
| Insurance/Risk | has_insurance | Master flag indicating whether the owner or business has any active insurance policy. |
| Insurance/Risk | medical_insurance / motor_vehicle_insurance / funeral_insurance | Specific types of insurance coverage possessed by the owner or business. |
| Risk Perception | perception_insurance_important | The owner's belief about the usefulness of insurance. |
| Risk Perception | perception_cannot_afford_insurance | Indicates cost as the main barrier to obtaining insurance. |
| Psychological | attitude_stable_business_environment | Confidence in the stability of the local business environment. |
| Psychological | attitude_satisfied_with_achievement | Personal satisfaction with business progress to date. |
| Psychological | attitude_more_successful_next_year | Optimism about stronger business performance in the coming year. |

## DATA PREPROCESSING AND FEATURE ENGINEERING
```mermaid
flowchart LR
    %% Data Ingestion & Prep
    Data[(Raw Survey Data)] --> FE[Feature Engineering Pipeline]
    
    %% Phase 1: Raw Baseline Track
    FE --> P1[Train Phase 1 Baseline Models]
    P1 --> Opt1{Optuna Weights for the TOP 3 models}
    Opt1 --> RawProbs(Blended Raw Probabilities)
    
    %% Phase 1 Output
    RawProbs -->|Argmax| Sub1[\Baseline Submission CSV<br>Classes: Low/Med/High/]
    
    %% Phase 2: Cleanlab Track
    FE --> CL[Cleanlab Label Purifier]
    CL -->|Drop Noise| P2[Train Phase 2 Purified Models]
    P2 --> Opt2{Optuna Weights for the TOP 3 models}
    Opt2 --> CleanProbs(Blended Clean Probabilities)
    
    %% Final Hedged Fusion Track
    RawProbs --> Hedge{Hedged Blend<br>e.g., 50% Raw / 50% Clean}
    CleanProbs --> Hedge
    
    %% Final Output
    Hedge -->|Argmax| Sub2[\Final Hedged Submission CSV<br>Classes: Low/Med/High/]

    %% High-Contrast Styling (Dark Text on Light Backgrounds)
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000;
    classDef highlight fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef accent fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000;
    classDef file fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#000;
    classDef target fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px,color:#000;
    
    class P1,P2,Opt1,Opt2 highlight;
    class CL,Hedge accent;
    class Sub1 file;
    class Sub2 target;
```
### 1. Robust Data Sanitization
* **Country-Aware Winsorization:** Because Eswatini, Lesotho, Malawi, and Zimbabwe have vastly different currencies and inflation rates, extreme financial outliers were capped at the 99th percentile *per country*. This neutralized data-entry typos without erasing legitimate high-earning businesses.
* **Logical Bounding:** Corrected impossible survey contradictions automatically (e.g., ensuring a business's age could not mathematically exceed the owner's working age).

### 2. Feature Engineering & Unsupervised Extraction
* **Domain-Driven Base Signals:** Translated raw survey text into mathematical indices. Created custom metrics like the `Master_Formalization_Index` (tracking tax and record-keeping compliance) and `Liquidity_Distress_Index` (multiplying high cash burn rates by negative psychological outlooks).
* **Unsupervised Pattern Extraction (K-Means):** Applied log transformations to revenue and expenses, then used a K-Means clustering algorithm to group businesses into distinct "Financial Maturity Tiers," allowing the gradient boosters to immediately recognize scale.
* **MICE Imputation:** Instead of filling missing continuous variables with a naive median, the pipeline uses Scikit-Learn's `IterativeImputer` (Multivariate Imputation by Chained Equations) to mathematically predict and fill missing values based on the business's other characteristics.

### 3. Handling Imbalance & Multicollinearity
* **Custom SMOTE Ratios:** To combat the severe class imbalance, Synthetic Minority Over-sampling Technique (SMOTE) was used. However, instead of a naive 1:1:1 balance, custom ratios were applied to gently boost the ultra-rare "High" class without flooding the dataset with synthetic noise.
* **Dynamic Correlation Drop:** A custom transformer automatically calculates a correlation matrix during training and drops heavily collinear features (Pearson > 0.90). This reduces dimensionality and prevents tree-based models from overfitting on redundant information.

### 4. Confident Learning 
* **Cleanlab Label Purification:** Survey data is full of human error. By using a highly constrained, shallow LightGBM "Judge," the pipeline flags rows where the Out-Of-Fold probability violently disagrees with the human label. By *dropping* these mathematically improbable rows (rather than attempting to relabel them), the final models learn on a 100% purified, contradiction-free dataset.

## MODEL EVALUATION AND SELECTION PIPELINE
### 1. Stratified Validation & Early Stopping
Models are not trained blindly for a set number of epochs. Evaluation relies on a strict **Stratified 80/20 Validation Split**, ensuring the exact distribution of Low/Medium/High classes is preserved in the holdout set. Furthermore, **Early Stopping** is actively monitored on the validation set. Once a model stops improving its Out-Of-Fold F1-Score, training is halted, the exact optimal tree count is locked in, and the model is refit on 100% of the data to prevent data leakage and overfitting.

### 2. The Evaluation Metric: Weighted F1-Score
The primary optimization metric for this pipeline is the **Weighted F1-Score**. Because the dataset suffers from severe class imbalance (the vast majority of SMEs are in the "Low" health tier), relying on standard Accuracy would be highly misleading. The Weighted F1-Score calculates the harmonic mean of precision and recall for each class and weights them by their actual support in the data, strictly penalizing models that lazily guess the majority class.

### 3. The Dual-Pipeline Strategy (Standard vs. Patterns)
For every base algorithm tested (**Random Forest**, **Extra Trees**, **XGBoost**, and **CatBoost**), the pipeline automatically generates and evaluates two distinct variations:
* **The "Standard" Pipeline:** Feeds the model the cleaned data alongside the core domain ratios (e.g., Burn Rate, Formalization Index).
* **The "Patterns" Pipeline:** Injects the highly complex, unsupervised features (e.g., K-Means Maturity Clusters, High-Value Interactions).
*(By testing both, the ensemble engine can later blend a "Standard" model that learned broad macroscopic trends with a "Patterns" model that captured hyper-specific edge cases).*

### 4.Ensembling Strategy: Optuna-Optimized Soft Voting

To generate the final predictions, the pipeline moves away from relying on a single algorithm and instead utilizes a highly optimized **Weighted Soft Voting Ensemble**. 

#### 4.1. Top-K Filtering (Preventing Ensemble Dilution)
A common pitfall in machine learning is averaging *all* trained models together, which allows weaker algorithms to drag down the accuracy of the best ones. To prevent this, the pipeline strictly enforces a **Top-K cutoff**. Only the elite, top-performing models (based strictly on Out-Of-Fold weighted F1 validation scores) are granted voting rights in the final ensemble.

#### 4.2. Optuna Weight Optimization
Instead of using a naive simple average (e.g., giving LightGBM, XGBoost, and CatBoost an equal 33% say), the engine leverages **Optuna (Tree-structured Parzen Estimator)**. Optuna runs hundreds of trials on the validation probabilities to discover the mathematically perfect fractional weights. For example, if LightGBM captured a vital macroeconomic trend, Optuna might dynamically assign it 55% voting power, while relegating CatBoost to 15%. 

#### 4.3. Probability Fusion & Final Argmax
Once the "Golden Weights" are discovered, the pipeline extracts the raw continuous probabilities (confidence levels) from the Top K models on the unseen Test Set. These probabilities are multiplied by their respective Optuna weights and stacked together. Finally, the default `Argmax` function collapses this fused probability matrix into the final discrete predictions (`Low`, `Medium`, `High`), yielding a submission that is significantly more robust than any individual model could achieve alone.

### 4.4. The Hedged Fusion (Raw + Purified Blending)
The final step of the pipeline mitigates the risks of both underfitting and overfitting to noise. We generate two separate Optuna-weighted ensembles: one trained on the **Raw Data** and one trained on the **Cleanlab Purified Data**. By fusing their probabilities together (e.g., a 50% Raw / 50% Clean split), we create a "Hedged Ensemble." The Raw models act as a grounded anchor to the true, messy real-world distribution, while the Purified models act as a precise mathematical corrector, pulling the predictions back into bounds when the raw models become overconfident on tricky, noisy edge cases.

## 5. MODEL PERFORMANCE
### 5.1 Top 3 Models before cleaning
This was trained on the full dataset (9618 rows)
| Model |  Ensemble Weight | F1 Weighted |
|---|---|---:|
| RandomForest_Patterns  | 0.6984 | 88.9182 | 
| ExtraTrees_Standard  | 0.0005 | 88.7962 | 
| ExtraTrees_Patterns | 0.3011 | 88.7536 |

### 5.1 Top 3 Models on cleaned data ( Out of the original 9618 rows, 638 rows were dropped)
This was trained on 8,980 rows
| Model |  Ensemble Weight | F1 Weighted |
|---|---|---:|
| RandomForest_Patterns  | 0.6243 | 93.0107 | 
| ExtraTrees_Standard  | 0.1795 | 93.2664 | 
| ExtraTrees_Patterns | 0.1962 | 92.9850 |  

### 5.4 Submission File Performance on Leaderboard
| Submission | Public Leaderboard | Private Leaderboard |
|---|---:|---:|
| Base Blended Submission | 89.2907738 | 88.4738645 |
| Base+cleaned Blended Submission  | 89.4169985 | 88.7214611 |

## 6.DISCUSSION AND CONCLUSION




