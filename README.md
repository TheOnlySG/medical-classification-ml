# SymptoSense

SymptoSense is a Machine Learning based medical screening web application focused on predicting diseases using patient medical data and generating visual health analysis reports.

The project combines:
- Machine Learning
- Medical analytics
- Flask web development
- Interactive screening forms
- Report generation concepts

---

# Diseases Currently Supported

- Heart Disease
- Diabetes
- Breast Cancer

---

# Project Progress

## Completed

### Machine Learning & Research
- Complete EDA for all 3 datasets
- Data preprocessing and cleaning
- Hidden missing value handling
- Correlation analysis
- Feature importance analysis
- Model evaluation and comparison
- Final model selection for deployment

### Finalized Models
| Disease | Final Model |
|---|---|
| Heart Disease | Random Forest Classifier |
| Diabetes | Random Forest Classifier |
| Breast Cancer | Logistic Regression |

### Production ML Pipeline
- Separate training scripts created
- Models exported using Joblib
- Scalers exported separately
- Organized project architecture

### Flask Frontend Development
- Multi-page Flask application setup
- Landing page completed
- Heart Disease screening form completed
- Diabetes screening form completed
- Breast Cancer screening form completed

---

# Breast Cancer Optimization

The Breast Cancer model was optimized using feature selection.

- Reduced from 30 features → 10 important features
- Accuracy remained ~97%
- Improved usability and frontend experience significantly

Selected features include:
- Radius
- Texture
- Perimeter
- Area
- Concavity
- Concave Points

---

# Tech Stack

## Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn

## Data Visualization
- Matplotlib
- Seaborn

## Web Development
- Flask
- HTML
- Bootstrap 5

## Model Serialization
- Joblib

---

# Current Status

🚧 Active Development

### Completed
- EDA
- Model training
- Model exporting
- Flask app setup
- Frontend screening forms

### Currently Working On
- Backend prediction pipelines
- Dynamic report pages
- Flask form handling
- Model integration with frontend

---

# Planned Features

- Real-time disease prediction
- Medical style analytics reports
- Dynamic graphs and comparisons
- Probability based prediction insights
- PDF report export
- Improved UI/UX
- Deployment and hosting

---

# Future Improvements

- XGBoost implementation
- Hyperparameter tuning
- Cross Validation
- ROC-AUC analysis
- SHAP / explainable AI
- Advanced medical analytics dashboard

---

# Learning Outcomes

This project helped in understanding:
- Real-world ML workflows
- EDA and preprocessing
- Feature engineering and selection
- Model evaluation techniques
- Flask backend fundamentals
- Frontend + ML integration
- Converting notebooks into deployable systems

---
