#  Fraud Detection using Logistic Regression

##  Project Overview
This project builds a machine learning model to detect fraudulent transactions using Logistic Regression.  
It goes beyond accuracy and evaluates the model using multiple classification metrics to better understand real-world performance.

---

## 🎯 Objective
- Identify fraudulent transactions from given data
- Evaluate model performance using:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix
  - ROC-AUC Score
- Understand the impact of threshold tuning on model behavior

---

##  Key Learnings
- Accuracy alone is not reliable for imbalanced datasets  
- Precision helps reduce false positives (false alarms)  
- Recall helps detect more fraud cases (important in real-world scenarios)  
- Lowering threshold increases recall but may reduce precision  
- ROC-AUC gives an overall measure of model performance  

---

## 🛠️ Tech Stack
- Python 
- Pandas
- NumPy
- Scikit-learn
- VS Code

---

## ⚙️ Steps Performed
1. Created dataset using Pandas  
2. Split data into training and testing sets (80/20, stratified)  
3. Scaled features using StandardScaler  
4. Trained Logistic Regression model  
5. Generated predictions  
6. Evaluated using:
   - Accuracy
   - Precision
   - Recall
   - Confusion Matrix  
7. Applied threshold tuning (0.5 → 0.3)  
8. Calculated ROC-AUC score  

---

## 📊 Results
- **Accuracy:** 1.0  
- **Precision:** 1.0  
- **Recall:** 1.0  
- **ROC-AUC Score:** 1.0  

> ⚠️ Note: The model shows perfect performance because the dataset is small and simple.  
> In real-world scenarios, results are usually lower and require more tuning.

---

## 📂 Project Structure
## 📂 Project Structure
fraud-detection-metrics/
│── fraud_detection.py
│── README.md