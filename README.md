# End-to-End Machine Learning Classification on WineQT Dataset

## a. Problem Statement
Build, evaluate, and deploy five supervised machine learning classification models on the Wine Quality dataset to predict whether a wine sample meets high-quality standards based on 12 physicochemical properties, and serve the pre-trained models via an interactive Streamlit web application.

## b. Dataset Description
- **Dataset Source:** Wine Quality Dataset (WineQT.csv from Kaggle as training dataset)
- **Instance Size:** 1,143 instances (914 training rows, 229 held-out test rows)
- **Feature Size:** 12 numerical input features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, and engineered total_acidity; row index Id removed)
- **Target Variable:** quality (Binary classification: Good Quality >= 6 then 1, Standard Quality < 6 then 0)

## c. GitHub Repository Link
- **GitHub Repository:** https://github.com/rdutta23/ML_Assigment
- **Live Streamlit App:** https://mlassigment-rahuldutta2026aj05665.streamlit.app/

## d. Models Used & Comparison Table

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.7904 | 0.8487 |      0.822  |   0.7823 | 0.8017 | 0.5805 |
| Decision Tree          |     0.7467 | 0.7761 |      0.8056 |   0.7016 | 0.75   | 0.5007 |
| KNN                    |     0.7511 | 0.8483 |      0.7445 |   0.8226 | 0.7816 | 0.4973 |
| Naive Bayes            |     0.7336 | 0.7836 |      0.7838 |   0.7016 | 0.7404 | 0.4716 |
| Support Vector Machine |     0.8035 | 0.8586 |      0.8319 |   0.7984 | 0.8148 | 0.6063 |

## Observations on Model Performance

| ML Model Name                     | Observation about model performance                                                                                                                        |
|:----------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression               | Provides a strong linear baseline as alcohol and volatile acidity correlate monotonically with wine quality, but misses non-linear chemical interactions.  |
| Decision Tree                     | Shows lower generalization and AUC because orthogonal axis-aligned splits overfit local density and acidity thresholds.                                    |
| KNN                               | Achieves the highest AUC and Recall by weighting local neighborhoods by inverse Euclidean distance after z-score scaling.                                  |
| Naive Bayes                       | Records the lowest accuracy and MCC because correlated chemical features (fixed acidity, pH, citric acid, total_acidity) violate conditional independence. |
| Support Vector Machine            | Delivers the highest overall Accuracy, Precision, and MCC; the RBF kernel effectively separates non-linear physicochemical boundaries.                     |
| Overall, Winner for your dataset? | Support Vector Machine (highest Accuracy and MCC), with KNN as a close runner-up leading in AUC and F1.                                                    |
