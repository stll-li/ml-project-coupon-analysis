# ml-project-coupon-analysis
A machine learning project analyzing coupon redemption behavior using real-world Meituan data, focusing on data preprocessing, model training, reproducibility, and interpretability.

# Coupon Redemption Prediction

This project analyzes coupon redemption behavior using machine learning models.
It includes data preprocessing, feature engineering, model training, evaluation,
and interpretation.

## Project Motivation
This project aims to understand the factors influencing coupon redemption behavior in large-scale online platforms
and to build interpretable and reproducible machine learning models on real-world data.

Note: The dataset is derived from real-world transaction records of Meituan, one of the largest on-demand service platforms in China, focusing on coupon redemption behavior.

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

### Environment
- Python 3.9.13  
- Virtual environment (PyCharm-managed)  
- Dependencies listed in `requirements.txt`

### Steps
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Run EDA analysis:
   ```bash
   python src/eda_analysis.py
3. Train models:
   ```bash
   python src/train_model.py
4. Generate predictions:
   ```bash
   python src/predict.py
   
## Notes
- Input data should be placed under data/ (train.xlsx, predict.xlsx).
- All outputs (figures, models, tables) will be automatically saved under the outputs/ directory.
