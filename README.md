# Drug Sensitivity Prediction Using Genomics of Drug Sensitivity in Cancer (GDSC)

This project aims to predict the **LN(IC50)** values—indicative of drug response—for different cancer cell lines using genomic features. The ultimate goal is to support **personalized medicine**, identifying the most effective drug for each patient based on molecular profiling.

## 📂 Dataset

- **Source**: [GDSC Dataset on Kaggle](https://www.kaggle.com/datasets/samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc)
- **Content**:
  - Genomic features (mutations, gene expressions, CNVs)
  - Drug response measures (IC50, LN(IC50))
  - Cancer cell line annotations
  - Drug metadata

## 🎯 Objective

Predict the **LN(IC50)** for a given pair of:

- **Cancer cell line**
- **Drug**

This will allow the identification of effective treatments tailored to the genetic profile of a patient’s tumor.

## 🧪 Methodology

The project follows these steps:

### 1. **Data Preprocessing**

- Merge drug response data with genomic features
- Handle missing values
- Normalize continuous features (e.g., gene expression)
- One-hot encode categorical variables (e.g., mutation status)

### 2. **Exploratory Data Analysis (EDA)**

- Distribution of LN(IC50) values
- Drug and cell line frequency
- Correlation analysis between features and response

### 3. **Model Building**

We experiment with several machine learning and deep learning models:

- Linear Regression
- Random Forest Regressor
- XGBoost
- Neural Networks (via TensorFlow/Keras)

### 4. **Training and Evaluation**

- Data split: training/validation/test (e.g., 70/15/15)
- Loss function: Mean Squared Error (MSE)
- Metrics: R² Score, RMSE, MAE
- K-Fold Cross-validation for robustness

### 5. **Hyperparameter Tuning**

- Grid Search / Random Search / Bayesian Optimization
- Dropout rates, learning rate, batch size, tree depth (depending on model)

### 6. **Interpretability**

- Feature importance (e.g., SHAP, permutation importance)
- Analysis of most influential genomic markers

## 🧠 Tools and Technologies

- Python 3.10+
- Pandas / NumPy
- scikit-learn
- TensorFlow / Keras
- XGBoost
- Matplotlib / Seaborn
- SHAP (for interpretability)

## 🔮 Example Use Case

Given a patient’s tumor characterized by specific mutations, expression levels, and CNV patterns, our model can:

- Predict the LN(IC50) value for several drugs
- Recommend the drug with the **lowest predicted LN(IC50)** (most sensitive)
