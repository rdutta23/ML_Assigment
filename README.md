# End-to-End Machine Learning Classification on WineQT Dataset

## a. Problem Statement
Build, evaluate, and deploy five supervised machine learning classification models on the Wine Quality dataset to predict whether a wine sample meets high-quality standards based on 12 physicochemical properties, and serve the pre-trained models via an interactive Streamlit web application.

## b. Dataset Description
- **Dataset Source:** Wine Quality Dataset (WineQT.csv from Kaggle / UCI Repository)
- **Instance Size:** 1,143 instances (914 training rows, 229 held-out test rows)
- **Feature Size:** 12 numerical input features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, and engineered total_acidity; row index Id removed)
- **Target Variable:** quality (Binary classification: 1 for Good Quality >= 6, 0 for Standard Quality < 6)

## c. GitHub Repository Link
- **GitHub Repository:** https://github.com/YOUR-USERNAME/YOUR-REPO-NAME
- **Live Streamlit App:** https://YOUR-APP-NAME.streamlit.app

## d. Models Used & Comparison Table

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.7511 | 0.828  |      0.7597 |   0.7903 | 0.7747 | 0.4974 |
| Decision Tree          |     0.738  | 0.7785 |      0.754  |   0.7661 | 0.76   | 0.4717 |
| KNN                    |     0.7642 | 0.8649 |      0.7431 |   0.8629 | 0.7985 | 0.5265 |
| Naive Bayes            |     0.7118 | 0.7904 |      0.7266 |   0.75   | 0.7381 | 0.4181 |
| Support Vector Machine |     0.7773 | 0.8518 |      0.7874 |   0.8065 | 0.7968 | 0.5507 |

## Observations on Model Performance

| ML Model Name                     | Observation about model performance                                                                                                                        |
|:----------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression               | Provides a strong linear baseline as alcohol and volatile acidity correlate monotonically with wine quality, but misses non-linear chemical interactions.  |
| Decision Tree                     | Shows lower generalization and AUC because orthogonal axis-aligned splits overfit local density and acidity thresholds.                                    |
| KNN                               | Achieves the highest AUC and Recall by weighting local neighborhoods by inverse Euclidean distance after z-score scaling.                                  |
| Naive Bayes                       | Records the lowest accuracy and MCC because correlated chemical features (fixed acidity, pH, citric acid, total_acidity) violate conditional independence. |
| Support Vector Machine            | Delivers the highest overall Accuracy, Precision, and MCC; the RBF kernel effectively separates non-linear physicochemical boundaries.                     |
| Overall, Winner for your dataset? | Support Vector Machine (highest Accuracy and MCC), with KNN as a close runner-up leading in AUC and F1.                                                    |
