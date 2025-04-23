# mini-project

Fraud Detection using U-Net Architecture for Tabular Data

This mini project implements a novel approach to credit card fraud detection using a **U-Net inspired neural network** tailored for tabular data. The model is trained in an unsupervised fashion on normal transactions and leverages reconstruction error to detect anomalies. Key features include:

- Custom **U-Net architecture** adapted for tabular input
- Trained exclusively on non-fraud cases to model normal transaction patterns
- **Reconstruction-based anomaly detection**
- **SMOTE** applied to validation and test sets for balanced evaluation
- Detailed metrics: AUC, F1, TPR/FPR, Confusion Matrix, and ROC curve visualization

**Dataset used**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

Libraries: TensorFlow, scikit-learn, imbalanced-learn, Matplotlib, Seaborn

