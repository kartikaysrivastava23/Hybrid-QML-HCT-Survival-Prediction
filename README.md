# Hybrid-QML-HCT-Survival-Prediction

This repository contains a hybrid quantum-inspired machine learning framework for predicting post-Hematopoietic Cell Transplantation (HCT) survival. The project explores the use of a quantum-inspired feature map to enhance a classical machine learning model, comparing its performance and fairness against traditional models like Logistic Regression and Random Forest. The primary goal is to achieve high predictive accuracy while ensuring equitable performance across different demographic groups.

## Overview

The project tackles the challenge of predicting event-free survival (EFS) for HCT patients using a large clinical dataset. A key contribution is the implementation of a hybrid model that combines a classical Logistic Regression classifier with a "Quantum-Inspired Feature Map." This feature map expands the dimensionality of the numerical features using trigonometric functions, a technique inspired by quantum feature encoding.

The analysis demonstrates that this hybrid approach not only improves overall predictive performance (measured by ROC-AUC) but also reduces performance disparity across sensitive attributes such as race and sex, leading to a more fair and robust model.

## Dataset

This study uses data from the CIBMTR (Center for International Blood and Marrow Transplant Research) Kaggle competition. Due to licensing and size constraints, the dataset is not included in this repository.

To reproduce the experiments, please follow these steps:
1.  Download the dataset from the [CIBMTR - HCT Outcome Prediction Kaggle Competition page](https://www.kaggle.com/competitions/cibmtr-hct-outcome-prediction/data).
2.  Place the following files into the `data/` directory of this repository:
    *   `train.csv`
    *   `test.csv`
    *   `data_dictionary.csv`

## Methodology

The core of the project is a comparative analysis between classical machine learning models and a novel hybrid quantum-inspired model.

### 1. Preprocessing

A standard preprocessing pipeline is applied to the data:
*   **Numerical Features**: Missing values are imputed using the median, followed by standardization using `StandardScaler`.
*   **Categorical Features**: Missing values are imputed with the most frequent value, and then features are one-hot encoded.

### 2. Models
Three models are trained and evaluated:

1.  **Logistic Regression**: A baseline classical model using the preprocessed data.
2.  **Random Forest**: A more complex ensemble model to serve as a strong classical baseline.
3.  **Hybrid Quantum-Inspired Model**: This model enhances the preprocessing pipeline for numerical features with a `QuantumInspiredFeatureMap` before feeding the data into a Logistic Regression classifier. The feature map transforms each scaled numerical feature `x` into a two-dimensional vector `[sin(πx), cos(πx)]`.

### 3. Evaluation
Models are evaluated on two main criteria:
*   **Overall Performance**: Measured by the Area Under the Receiver Operating Characteristic Curve (ROC-AUC) on a held-out validation set.
*   **Fairness**: Assessed by calculating the ROC-AUC for subgroups based on `race` and `sex`. The disparity is quantified as the difference between the maximum and minimum ROC-AUC scores across the groups.

## Results

The hybrid quantum-inspired model demonstrated superior performance and fairness compared to the classical Random Forest baseline.

| Model                     | Overall ROC-AUC | Race Disparity (Max-Min AUC) | Sex Disparity (Max-Min AUC) |
| ------------------------- | --------------- | ---------------------------- | --------------------------- |
| Random Forest             | 0.746           | 0.0350                       | 0.0301                      |
| Hybrid Quantum-Inspired   | **0.817**       | **0.0348**                   | **0.0170**                  |

### Overall Performance

The hybrid model achieves a significantly higher ROC-AUC, indicating better overall predictive power.

![Overall ROC Curves](https://raw.githubusercontent.com/kartikaysrivastava23/Hybrid-QML-HCT-Survival-Prediction/main/figures/figure1_roc_curves.png)

### Fairness Analysis: Race

While both models show similar levels of performance disparity across racial groups, the hybrid model consistently achieves a much higher ROC-AUC for every single group.

![ROC-AUC by Race Group](https://raw.githubusercontent.com/kartikaysrivastava23/Hybrid-QML-HCT-Survival-Prediction/main/figures/figure2_race_auc.png)

### Fairness Analysis: Sex Match

The hybrid model not only improves performance across all sex-match categories but also reduces the performance disparity by nearly half compared to the Random Forest model.

![ROC-AUC by Sex Group](https://raw.githubusercontent.com/kartikaysrivastava23/Hybrid-QML-HCT-Survival-Prediction/main/figures/figure4_sex_auc.png)

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kartikaysrivastava23/Hybrid-QML-HCT-Survival-Prediction.git
    cd Hybrid-QML-HCT-Survival-Prediction
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up the dataset:**
    Download the data from Kaggle and place `train.csv`, `test.csv`, and `data_dictionary.csv` inside the `data/` folder as described [above](#dataset).

4.  **Run the notebook:**
    Open and run the `notebook/Hybrid_QML_HCT_Experiments.ipynb` notebook in a Jupyter environment to replicate the analysis and results.

## Requirements

The project dependencies are listed in `requirements.txt`:
*   numpy
*   pandas
*   scikit-learn
*   matplotlib
