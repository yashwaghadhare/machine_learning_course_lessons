# Machine Learning Lessons

# Lesson 1: Course Introduction

## 1.1 Overview
This repository contains structured lessons on Machine Learning (ML), covering foundational concepts, supervised and unsupervised learning, regression, classification, ensemble methods, and practical Python examples using popular ML libraries.

## 1.2 Table of Contents
- [Lesson 2: Introduction to Machine Learning](#lesson-2-introduction-to-machine-learning)
- [Lesson 3: Supervised Learning](#lesson-3-supervised-learning)
- [Lesson 4: Regression and Applications](#lesson-4-regression-and-applications)
- [Lesson 5: Classification and Applications](#lesson-5-classification-and-applications)
- [Lesson 6: Unsupervised Algorithms](#lesson-6-unsupervised-algorithms)
- [Lesson 7: Ensemble Learning](#lesson-7-ensemble-learning)

---

# Lesson 2: Introduction to Machine Learning

## 2.1 Introduction
Machine Learning (ML) is a branch of AI that lets computers learn patterns from data to make predictions or decisions without being explicitly programmed.

**Applications**:  
- Spam detection  
- Image recognition  
- Stock prediction  
- Recommendation systems  

## 2.2 What Is Machine Learning?
ML uses data to train models, which can then make predictions on new data.

**Steps**:  
1. Collect data  
2. Clean & prepare data  
3. Train a model  
4. Evaluate performance  
5. Make predictions  

**Python Packages**: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tensorflow`, `keras`

**Simple Code Example**:
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

data = load_iris()
X, y = data.data, data.target

model = DecisionTreeClassifier()
model.fit(X, y)
print(model.predict([X[0]]))  # Predict for first sample
```

## 2.3 Types of Machine Learning
- **Supervised Learning**: Learns from labeled data (e.g., classification, regression)  
- **Unsupervised Learning**: Finds patterns in unlabeled data (e.g., clustering, PCA)  
- **Reinforcement Learning**: Learns via trial and error using rewards (e.g., games, robotics)  

## 2.4 ML Pipeline & MLOps
- **Pipeline**: Data → Features → Model → Evaluate → Deploy → Monitor  
- **MLOps**: Applying DevOps practices to manage ML models in production.

## 2.5 Python Packages Overview
- `numpy`, `pandas` → Data handling  
- `matplotlib`, `seaborn` → Visualization  
- `scikit-learn` → ML algorithms  
- `tensorflow`, `keras` → Deep learning  

---

# Lesson 3: Supervised Learning

## 3.1 Introduction
Supervised Learning is a type of machine learning where the model learns from labeled data—data with input-output pairs.  
The goal is to predict the output for new, unseen inputs.

**Applications**:  
- Predicting house prices (regression)  
- Email spam detection (classification)  
- Customer churn prediction  

## 3.2 Supervised Learning
**Two main types**:  
- **Regression**: Predicts continuous values (e.g., price, temperature)  
- **Classification**: Predicts discrete categories (e.g., spam/ham, disease/no disease)  

## 3.3 Preparing and Shaping Data
- Clean missing values  
- Encode categorical variables  
- Split into training and test sets  
- Scale or normalize features if needed  

## 3.4 Overfitting and Underfitting
- **Overfitting**: Model learns training data too well, performs poorly on new data  
- **Underfitting**: Model is too simple, performs poorly on training and test data  

**Prevention**:  
- Use more data  
- Simplify or regularize the model  
- Use cross-validation  

## 3.5 Regularization
Adds penalty for complex models to reduce overfitting.  
**Common techniques**: Lasso (L1), Ridge (L2)

## 3.6 Small Code Example
```python
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

data = load_boston()
X, y = data.data, data.target

model = LinearRegression()
model.fit(X, y)
print(model.predict([X[0]]))  # Predict for first sample
```
> **Note**: `load_boston` is deprecated; use another dataset in practice.

---

# Lesson 4: Regression and Applications

## 4.1 Introduction
Regression is a type of supervised learning where the goal is to predict continuous values based on input features.

**Applications**:  
- Predicting house prices  
- Forecasting sales or stock prices  
- Estimating temperatures  

## 4.2 Types of Regression
- **Linear Regression**: Predicts output as a straight-line function of input.  
- **Polynomial Regression**: Captures non-linear relationships using higher-degree terms.  
- **Ridge Regression**: Adds L2 penalty to prevent overfitting.  
- **LASSO Regression**: Adds L1 penalty for feature selection and regularization.  
- **Logistic Regression**: Used for classification (predicts probability of a class).  

## 4.3 Linear Regression Example
```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

data = load_diabetes()
X, y = data.data, data.target

model = LinearRegression()
model.fit(X, y)
print(model.predict([X[0]]))  # Predict first sample
```

## 4.4 Logistic Regression Example (Classification)
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

data = load_iris()
X, y = data.data, data.target

model = LogisticRegression(max_iter=200)
model.fit(X, y)
print(model.predict([X[0]]))  # Predict first sample
```

## 4.5 Key Concepts
- **Critical assumptions for linear regression**: Linearity, independence, homoscedasticity, normality of errors.  
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Handles class imbalance by creating synthetic samples.  
- **Model Evaluation**: Use metrics like RMSE (for regression) and accuracy or F1-score (for classification).  

---

# Lesson 5: Classification and Applications

## 5.1 Introduction
Classification is a type of supervised learning where the goal is to predict discrete categories.

**Applications**:  
- Spam detection (spam/ham)  
- Disease diagnosis (disease/no disease)  
- Customer churn (yes/no)  

## 5.2 Types of Classification
- **Binary Classification**: Two classes (e.g., yes/no)  
- **Multi-class Classification**: More than two classes (e.g., iris species)  
- **Multi-label Classification**: Predict multiple labels for each sample  

## 5.3 Key Concepts
- **Performance metrics**: Accuracy, Precision, Recall, F1-score, Cohen’s Kappa  
- **Overfitting prevention**: Cross-validation, regularization, pruning trees, etc.  
- **Feature selection**: Techniques like Boruta help select important features  

## 5.4 Common Classification Algorithms & Tiny Snippets
1. **Naive Bayes**:
    ```python
    from sklearn.datasets import load_iris
    from sklearn.naive_bayes import GaussianNB

    X, y = load_iris(return_X_y=True)
    model = GaussianNB()
    model.fit(X, y)
    print(model.predict([X[0]]))
    ```
2. **Decision Tree**:
    ```python
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(X, y)
    print(model.predict([X[0]]))
    ```
3. **Random Forest**:
    ```python
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X, y)
    print(model.predict([X[0]]))
    ```
4. **K-Nearest Neighbors (KNN)**:
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    print(model.predict([X[0]]))
    ```
5. **Support Vector Machine (SVM)**:
    ```python
    from sklearn.svm import SVC
    model = SVC()
    model.fit(X, y)
    print(model.predict([X[0]]))
    ```

---

# Lesson 6: Unsupervised Algorithms

## 6.1 Introduction
Unsupervised learning finds patterns or structures in unlabeled data. No output labels are provided; the algorithm tries to learn the underlying structure.

**Applications**:  
- Customer segmentation  
- Anomaly detection  
- Market basket analysis  
- Dimensionality reduction for visualization  

## 6.2 Types of Unsupervised Algorithms
- **Clustering**: Groups similar data points together (e.g., K-Means, Hierarchical)  
- **Association**: Finds rules between variables (e.g., market basket analysis)  
- **Dimensionality Reduction**: Reduces number of features while retaining important info (e.g., PCA, SVD, ICA)  
- **Outlier Detection**: Detects anomalies in data (e.g., PyOD library)  

## 6.3 Clustering Examples

### K-Means
```python
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

X, _ = load_iris(return_X_y=True)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
print(kmeans.labels_)  # Cluster labels for each sample
```

### Hierarchical Clustering
```python
from scipy.cluster.hierarchy import linkage, fcluster

Z = linkage(X, 'ward')
labels = fcluster(Z, 3, criterion='maxclust')
print(labels)
```

## 6.4 Dimensionality Reduction

### PCA (Principal Component Analysis)
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(X_reduced[:5])
```

### SVD (Singular Value Decomposition)
```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)
print(X_svd[:5])
```

## 6.5 Outlier Detection
```python
from pyod.models.knn import KNN

clf = KNN()
clf.fit(X)
outliers = clf.labels_
print(outliers[:10])  # 0 = normal, 1 = outlier
```

---

# Lesson 7: Ensemble Learning

## 7.1 Introduction
Ensemble learning combines multiple models to improve overall performance. The idea: “Two heads are better than one.”

**Applications**:  
- Fraud detection  
- Loan approval prediction  
- Image classification  

## 7.2 Types of Ensemble Learning
- **Bagging (Bootstrap Aggregating)**: Multiple models trained on random subsets of data; results are averaged or voted.  
- **Boosting**: Models trained sequentially, each focusing on mistakes of the previous one.  
- **Stacking**: Combines predictions of multiple models using a meta-model.  

## 7.3 Bagging Example
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=5)
model.fit(X, y)
print(model.predict([X[0]]))
```

## 7.4 Boosting Example
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=5)
model.fit(X, y)
print(model.predict([X[0]]))
```

## 7.5 Stacking Example
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

estimators = [('dt', DecisionTreeClassifier()), ('svc', SVC())]
stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack_model.fit(X, y)
print(stack_model.predict([X[0]]))
```

## 7.6 Key Concepts
- Ensemble methods reduce errors and improve robustness.  
- Bagging reduces variance, Boosting reduces bias, Stacking combines strengths of different models.  
- Works well with weak learners (models slightly better than random).  

