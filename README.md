# ml-project-coupon-analysis
A machine learning project analyzing coupon redemption behavior using real-world Meituan data, focusing on data preprocessing, model training, reproducibility, and interpretability.

# Coupon Redemption Prediction

This project analyzes coupon redemption behavior using machine learning models.
It includes data preprocessing, feature engineering, model training, evaluation,
and interpretation.

## Project Motivation
This project aims to understand the factors influencing coupon redemption behavior 
and to build reproducible and interpretable machine learning models on real-world data.

## Methodology
- Data cleaning with explicit business rules (e.g., handling zero-denominator ratios)
- Feature engineering (e.g., coupon value-to-threshold ratio)
- Multiple classification models (Random Forest, XGBoost, LightGBM, etc.)
- Fixed random seeds and stratified cross-validation for reproducibility
- Model interpretability using feature importance and SHAP analysis

## Reproducibility
- Python 3.9.13
- virtualenv-based isolated environment
- Fixed random seeds across data split, CV, and model training
- All experiments executed via Jupyter Notebook within a PyCharm-managed environment

## Project Structure
- data/: raw and schema files
- notebooks/: exploratory analysis
- src/: training pipeline
- outputs/: figures and model results

## Models Used
- KNN
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Gradient Boosting
- AdaBoost

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Run:
   python src/model_pipeline.py

## Notes
- Dataset not included due to privacy.
- All results are reproducible with provided scripts.
