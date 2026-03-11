#  Student Performance Prediction

A Machine Learning project that predicts students’ exam scores based on academic, behavioral, and productivity-related features.  
The project applies multiple regression models and compares their performance to select the best-performing algorithm.

---

##  Overview

This project focuses on predicting student academic performance using supervised machine learning techniques.

We:

- Performed data preprocessing  
- Conducted Exploratory Data Analysis (EDA)  
- Applied feature selection  
- Trained multiple regression models  
- Compared performance using evaluation metrics  
- Analyzed feature importance  

The goal is to identify the most impactful factors affecting student exam scores and build a high-performing predictive model.

---

##  Technical Highlights

- Regression-based ML modeling  
- K-Fold Cross Validation  
- Model comparison using R² Score, MAE, and MSE  
- Feature importance analysis using Random Forest  
- Data preprocessing including:
  - Handling missing values  
  - Removing misleading values  
  - Encoding categorical variables  
  - Feature scaling  

---

##  Installation

### 1️ Clone the repository

```bash
git clone https://github.com/majedalsulami/Student-performance-prediction.git
cd Student-performance-prediction
```

### 2️ Install required libraries

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm catboost kagglehub tqdm
```

---

##  Usage

Download the dataset (automatically handled using KaggleHub).

Run the notebook:

```bash
jupyter notebook Student_Performance_Prediction.ipynb
```

Execute cells in order:

- Import libraries  
- Load dataset  
- Perform EDA  
- Train models  
- Compare results  
- Analyze feature importance  

---

##  Libraries and Technologies

- Python  
- Pandas – Data manipulation  
- NumPy – Numerical computing  
- Matplotlib & Seaborn – Data visualization  
- Scikit-learn – Machine learning models & evaluation  
- LightGBM – Gradient boosting framework  
- CatBoost – Categorical boosting algorithm  
- KaggleHub – Dataset downloading  

---

##  Dataset

Dataset downloaded from Kaggle: [student-performance-dataset](https://www.kaggle.com/datasets/amar5693/student-performance-dataset)


File used:

```
ultimate_student_productivity_dataset_5000.csv
```

###  Target Variable

```
exam_score
```

###  Example Features

- Gender  
- Academic level  
- Internet quality  
- Productivity score  
- Study-related attributes  

### Data Cleaning Included

- Removing misleading values (e.g., "other" in gender column)  
- Handling missing values  
- Removing duplicates  

---

##  Analysis Workflow

### 1️ Data Loading

- Dataset downloaded using KaggleHub  
- Loaded into Pandas DataFrame  

### 2️ Data Cleaning

- Checked duplicates  
- Checked missing values  
- Removed misleading categorical entries  

### 3️ Exploratory Data Analysis (EDA)

- Target distribution visualization  
- Statistical summaries  
- Feature inspection  

### 4️ Feature Engineering

- Label encoding categorical variables  
- Scaling numerical features (MinMaxScaler)  

### 5️ Model Training

Models used:

- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Support Vector Regressor (SVR)  
- Decision Tree Regressor  
- Random Forest Regressor  
- LightGBM Regressor  
- CatBoost Regressor  

Used:

- K-Fold Cross Validation  

### 6️ Model Evaluation

Evaluation Metrics:

- R² Score  
- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  

 Since R² was used for comparison:

Higher R² score = Better performance

### 7️ Best Model

 Random Forest Regressor achieved the highest R² score.

---

##  Key Features

- Multiple model comparison  
- Proper preprocessing pipeline  
- Cross-validation for reliable evaluation  
- Feature importance analysis  
- Clear visualization of results  

---

##  Feature Importance

Using Random Forest:

`productivity_score` was identified as the most important feature.

This indicates that productivity-related behaviors strongly influence exam performance.

---

## 📁 Project Structure

```
├── Student_Performance_Prediction.ipynb
├── README.md
├── requirements.txt

```

---

##  What I Learned

- How to compare multiple regression models  
- Importance of preprocessing before modeling  
- Why ensemble models often outperform linear models  
- How to interpret feature importance  
- Practical implementation of K-Fold Cross Validation  

---

##  Future Improvements
 
- Build a full ML pipeline  
- Deploy model using Flask or FastAPI  
- Create a simple web interface   


---

##  License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute this project.

---

##  Contributing

Contributions are welcome!

- Fork the repository  
- Create a new branch  
- Make improvements  
- Submit a Pull Request  

---

##  Questions

If you have any questions, suggestions, or feedback:

- Email: majedsaadalsulami@gmail.com 
- Github: [MajedAlsulami](https://github.com/MajedAlsulami)
---

 If you found this project useful, consider giving it a star!
