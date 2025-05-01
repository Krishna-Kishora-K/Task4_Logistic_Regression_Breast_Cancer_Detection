# Task 4: Logistic Regression - Breast Cancer Detection

## Objective:
Build a binary classifier using **Logistic Regression** to detect if a tumor is malignant (M) or benign (B).

---

## Tools Used:
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

- Outputs are attached
---

## Steps Performed:

1. **Load Data**  
   - Read the CSV file and dropped unnecessary columns.

2. **Preprocessing**  
   - Converted diagnosis 'M'/'B' to 1/0  
   - Checked for nulls

3. **Train-Test Split**  
   - Used 80% for training, 20% for testing

4. **Feature Scaling**  
   - Applied `StandardScaler`

5. **Model Training**  
   - Trained `LogisticRegression()` model

6. **Prediction & Evaluation**  
   - Used:
     - Confusion Matrix
     - Classification Report
     - ROC Curve & AUC Score

---

## Evaluation:

- **Precision, Recall, F1-score** printed via `classification_report`
- **ROC-AUC Score** plotted and calculated
- **Confusion Matrix** visualized with heatmap

---

## Sigmoid Function:

Logistic Regression uses the **Sigmoid function**:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

It maps any number to a value between **0 and 1**, representing the **probability** of class 1.


---

## How to Run:

1. Install libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
