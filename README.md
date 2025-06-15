# Cybersecurity-Threat-Detection-ML

This repository contains a machine learning project focused on detecting various cybersecurity threats using the NSL-KDD dataset. The project involves data preprocessing, feature engineering, and the application of multiple classification algorithms to identify different types of network intrusions.

## Project Goal

The primary goal of this project is to build and evaluate robust machine learning models capable of accurately classifying network traffic as benign or various types of attacks (e.g., DoS, U2R, R2L, Probe). This can aid in enhancing cybersecurity defenses by providing automated threat detection.

## Files Overview

* **`cybersecurity_threat_detection.ipynb`**: This Jupyter notebook details the entire machine learning pipeline, including:
    * Loading and initial exploration of the NSL-KDD dataset.
    * Data preprocessing steps such as handling categorical features, encoding, and scaling.
    * Training and evaluation of multiple classification models (e.g., Random Forest, Naïve Bayes, Decision Tree, Logistic Regression, AdaBoost, MLP).
    * Performance metrics (Accuracy, Precision, Recall, F1-Score, FPR, ROC AUC, Kappa, MAE) and confusion matrices are used for model comparison.
    * Hyperparameter tuning using GridSearchCV.
    * Analysis of model trade-offs, computation time, and business recommendations.

## Dataset

The project utilizes the **NSL-KDD dataset**, a widely used benchmark dataset for network intrusion detection. The dataset is fetched directly within the notebook from a public URL, ensuring reproducibility.

## Analysis Highlights

* **Comprehensive Model Evaluation**: Multiple machine learning algorithms were trained and rigorously evaluated using a diverse set of metrics.
* **Performance Comparison**: The models' performance was compared in terms of accuracy, precision, recall, F1-score, and computational efficiency to identify the most suitable algorithm for real-time deployment (e.g., Random Forest often shows strong overall performance).
* **Practical Recommendations**: Insights are provided on how different models' characteristics (e.g., precision vs. speed) influence their applicability in various cybersecurity contexts.

## Technologies Used

* Python
* Pandas (for data manipulation)
* NumPy (for numerical operations)
* Scikit-learn (for machine learning models and preprocessing)
* Matplotlib & Seaborn (for visualizations)

## Getting Started

To explore or run the analysis:

1.  Clone this repository.
2.  Ensure you have Python installed along with the necessary libraries (you can install them via `pip install pandas numpy scikit-learn matplotlib seaborn`).
3.  Open `cybersecurity_threat_detection.ipynb` in a Jupyter environment (e.g., Jupyter Lab, VS Code with Jupyter extension, or Google Colab).
4.  Run the cells sequentially to execute the data loading, preprocessing, model training, and evaluation steps. The dataset will be downloaded automatically by the notebook.

## Contact

For any questions or further information, please contact [Jeremiyah/jeremypeter016@gmail.com].
