# Customer Churn Prediction
End-to-end machine learning project to predict telecom customer churn and extract actionable retention insights.
Customer churn prediction using machine learning with preprocessing, model comparison, ROC-AUC analysis, feature importance, and tuning.

## Project Highlights
- Performs data exploration and preprocessing on customer behavior/service features.
- Trains and compares Logistic Regression and Random Forest models.
- Applies class balancing and threshold tuning for churn-focused recall.
- Uses ROC-AUC, confusion matrix, and cross-validation for robust evaluation.
- Tunes Random Forest hyperparameters with `GridSearchCV`.

## Repository Structure
- `CustomerChurn.ipynb`: Main notebook (EDA, modeling, evaluation, business insights)
- `data/customer_churn.csv`: Input dataset
- `requirements.txt`: Python dependencies

## How to Run
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook CustomerChurn.ipynb
   ```

## Problem Statement
Predict whether a customer will churn (`Churn = 1`) based on usage, complaints, plan, and profile features.

## Modeling Approach
- Baseline: Logistic Regression (`class_weight='balanced'`)
- Tree Model: Random Forest
- Tuning: Grid search over `n_estimators`, `max_depth`, and `min_samples_split`
- Validation: Stratified train/test split + 5-fold cross-validation (ROC-AUC)

## Business Value
The project helps retention teams identify at-risk customers and prioritize intervention campaigns, with threshold tuning to improve churn recall when missing a churner is costly.
