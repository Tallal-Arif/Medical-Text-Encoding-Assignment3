# 🧬 DS-Disease-Classification

> Assignment 3 for 22F-3675 - Data Science for Software Engineering  
> Comparing TF-IDF and One-hot Encoding for Disease Classification

## 📌 Objectives

- Compare the effectiveness of **TF-IDF** vs **One-hot encoding** on medical datasets
- Use **PCA** and **SVD** for dimensionality reduction
- Evaluate models like **KNN** (k=3,5,7) and **Logistic Regression**
- Analyze clustering patterns and performance metrics
- Deploy a **Streamlit app** for interactive prediction

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/DS-Disease-Classification.git
cd DS-Disease-Classification
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Run the Jupyter notebook
```bash
jupyter notebook notebook/Assignment3_Notebook.ipynb
```

### 4. Launch the Streamlit app
```bash
streamlit run streamlit_app/knn_app.py
```

---

## 📊 Results

- Logistic Regression and KNN failed to generalize due to one sample per disease.
- One-hot showed slightly tighter clustering in PCA/SVD plots.
- TF-IDF captured richer semantic signals, but had lower initial separability.

---

## 🧠 Critical Analysis

- TF-IDF captures importance; One-hot is more interpretable and clinical.
- Dataset is too sparse for effective supervised learning.
- Suggest augmenting data or using multi-label/multi-instance learning in the future.

---

## 📚 Deliverables

- [x] Jupyter Notebook (`.ipynb`)
- [x] Streamlit App (`.py`)
- [x] Results CSVs
- [x] Blog Post / LinkedIn Post *(optional)*

---

## 👨‍💻 Author

**Jerry** — [LinkedIn](https://linkedin.com/in/YOURHANDLE) | Roll: `22F-3675`
