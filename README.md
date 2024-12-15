# Project Description

## Overview

This repository implements machine learning algorithms on the UNSW-NB15 dataset, which contains network traffic data including various types of cyberattacks and normal activities. The project focuses on three key algorithms: K-Nearest Neighbors (KNN), Gaussian Naive-Bayes, and ID3, with both from-scratch and scikit-learn implementations.

## Key Features
- **KNN from Scratch**: Implementation of the K-Nearest Neighbors algorithm, capable of accepting two input parameters:
  - The number of neighbors.
  - The distance metric (Euclidean, Manhattan, or Minkowski).
  
- **Gaussian Naive-Bayes from Scratch**: Implementing Gaussian Naive-Bayes algorithm from scratch.

- **ID3 from Scratch**: Implementation of the ID3 algorithm, including numeric data preprocessing based on lecture materials.

- **Comparison**: A comparison between the from-scratch implementations and scikit-learn versions for KNN, Gaussian Naive-Bayes, and ID3. The scikit-learn ID3 is implemented using the `DecisionTreeClassifier` with `criterion='entropy'`.


## Steps to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/project-name.git
cd project-name
```

### 2. Run the Jupyter Notebook
```bash
cd src
jupyter notebook "IF3170 Artificial Intelligence _ Tugas Besar 2 Notebook Template.ipynb"
```

## Task Division

| **NIM**     | **Task**                                             |
|-------------|------------------------------------------------------|
| 13522043   | ID3 Modelling |
| 13522054   | KNN Modelling |
| 13522068   | Data Cleaning and Preprocessing |
| 13522118   | Gaussian Naive Bayes Modelling |
