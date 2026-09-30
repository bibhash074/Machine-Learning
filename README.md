# Machine Learning Experiment

## 📌 Project Overview

This project focuses on applying different machine learning algorithms to network traffic datasets.

The datasets are processed, cleaned, and prepared for machine learning. PCA (Principal Component Analysis) is used for dimensionality reduction, followed by training and evaluation of different machine learning models.

The performance of the models is evaluated using standard machine learning metrics such as Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.

---

## 🎯 Objectives

The main objectives of this project are:

* To understand the process of preparing a real-world dataset for machine learning.
* To preprocess and clean network traffic data.
* To apply PCA for dimensionality reduction.
* To implement different machine learning algorithms.
* To compare model performance using different test sizes.
* To evaluate the models using standard performance metrics.
* To understand the effect of different PCA component values on model performance.

---

## 📂 Dataset

The project uses network traffic datasets containing information related to different working days and web attacks.

The datasets used are:

```text
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
```

The datasets are combined and used for the machine learning experiments.

---

## 🧹 Data Preprocessing

The following preprocessing steps are performed on the dataset:

1. Loading the datasets.
2. Combining the datasets.
3. Cleaning column names.
4. Separating input features and target values.
5. Converting feature values into numeric form.
6. Replacing infinite values with missing values.
7. Handling missing values using mean imputation.
8. Encoding target labels into numerical values.

---

## 🔬 Principal Component Analysis (PCA)

PCA is used to reduce the number of features while retaining important information from the dataset.

The following PCA component values are used:

```text
2
4
6
8
9
```

Using different numbers of components helps in studying the effect of dimensionality reduction on model performance.

---

## 🤖 Machine Learning Algorithms

The following algorithms are used in the project:

### 1. Linear Regression

Linear Regression is used to study the relationship between input features and the target values.

### 2. Logistic Regression

Logistic Regression is used for classification of the target classes.

### 3. Ridge Regression

Ridge Regression is a regularized regression technique that uses L2 regularization.

### 4. Lasso Regression

Lasso Regression uses L1 regularization and can help reduce the effect of less important features.

### 5. Elastic Net

Elastic Net combines the properties of Lasso and Ridge regression.

---

## 🧪 Train-Test Split

The dataset is divided into training and testing portions.

The following test sizes are considered:

```text
20%
40%
60%
```

Different test sizes are used to observe how the amount of training data affects model performance.

---

## 📊 Evaluation Metrics

The models are evaluated using:

### Accuracy

Measures the overall percentage of correctly predicted samples.

### Precision

Measures how many of the samples predicted as a particular class are actually from that class.

### Recall

Measures how many actual samples of a class were correctly identified.

### F1 Score

F1 Score combines Precision and Recall into a single measure.

### Confusion Matrix

The confusion matrix shows the relationship between actual and predicted classes.

### Classification Report

A classification report provides detailed performance information for the classes.

---

## 📈 Results

For each experiment, the following information is recorded:

```text
Algorithm
Test Size
PCA Components
Accuracy
Precision
Recall
F1 Score
```

The results are saved for further analysis and comparison.

---

## 📁 Project Structure

```text
Machine-Learning-Project/
│
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│
├── project.py
│
├── 75_ML_Results/
│   ├── JPG result files
│   └── ALL_75_RESULTS.csv
│
└── README.md
```

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## ▶️ How to Run

### Step 1: Install Python

Make sure Python is installed on your computer.

### Step 2: Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Step 3: Add the Dataset

Place the required CSV datasets in the same project directory as the Python program.

### Step 4: Run the Program

```bash
python project.py
```

The results will be generated and saved in the output folder.

---

## 📋 Experiment Configuration

| Parameter      | Values                                   |
| -------------- | ---------------------------------------- |
| Algorithms     | 5                                        |
| Test Sizes     | 20%, 40%, 60%                            |
| PCA Components | 2, 4, 6, 8, 9                            |
| Evaluation     | Accuracy, Precision, Recall, F1 Score    |
| Visualization  | Confusion Matrix & Classification Report |

---

## 🎓 Conclusion

This project provides practical experience in applying machine learning techniques to network traffic data.

It demonstrates the complete process of preparing a dataset, reducing its dimensions using PCA, applying different machine learning algorithms, and evaluating their performance using standard metrics.

The project also helps in understanding how changes in PCA components and test-set size can affect machine learning results.

---

## 👨‍💻 Authors

Bibhash Raut
Abhinav Singh
Sumit Kumar

GitHub: bibhash074
---

## 📜 License

This project is created for educational purposes.
