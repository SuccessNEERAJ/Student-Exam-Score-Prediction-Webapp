# 🎓 Student Performance Prediction System

A comprehensive machine learning-powered web application built with Streamlit that analyzes student performance data to predict academic outcomes and identify at-risk students who need intervention.

## 🔗Website Link - https://student-exam-score-prediction-webapp.streamlit.app/

## 📋 Table of Contents

- [Overview](##overview)
- [Features](##features)
- [Screenshots](##screenshots)
- [Installation](##installation)
- [Usage](##usage)
- [Dataset Format](##dataset-format)
- [Technical Details](##technical-details)
- [Project Structure](##project-structure)
- [Contributing](##contributing)
- [License](##license)

## 🌟 Overview

The Student Performance Prediction System is an advanced analytics tool designed for educational institutions to:

- **📊 Analyze** student demographics and performance patterns
- **🤖 Predict** student scores using machine learning algorithms
- **⚠️ Identify** at-risk students who need immediate intervention
- **💡 Recommend** personalized interventions for student improvement
- **🔮 Forecast** individual student performance based on various factors

This system helps educators and administrators make data-driven decisions to improve student outcomes and provide targeted support where it's needed most.

## ✨ Features

### 🔍 Data Analysis & Visualization
- **Comprehensive Data Overview**: View dataset statistics, sample data, and data quality metrics
- **Interactive Visualizations**: Explore distributions of numeric and categorical variables
- **Score Analysis**: Detailed analysis of academic performance across different metrics

### 🤖 Machine Learning
- **Random Forest Regressor**: High-accuracy model for score prediction
- **Feature Engineering**: Automated creation of composite scores (Academic, Lifestyle, Support System, Engagement)
- **Model Performance Metrics**: R² Score, RMSE, MAE, and cross-validation results
- **Feature Importance Analysis**: Identify the most influential factors affecting student performance

### ⚠️ At-Risk Student Identification
- **Multi-Factor Risk Assessment**: Evaluate students based on 8 different risk criteria
- **Priority Scoring**: Rank students by risk level for targeted intervention
- **Demographic Analysis**: Risk patterns across departments, gender, and age groups

### 💡 Intervention Recommendations
- **Personalized Recommendations**: Tailored intervention strategies for each at-risk student
- **Academic Support**: Tutoring, exam preparation, and study skills recommendations
- **Lifestyle Guidance**: Stress management, sleep hygiene, and study schedule optimization
- **Support System Enhancement**: Financial aid, digital access, and engagement opportunities

### 🔮 Individual Prediction
- **Interactive Prediction Tool**: Input student data to predict performance
- **Real-time Risk Assessment**: Immediate identification of risk factors
- **Actionable Insights**: Specific recommendations based on predicted outcomes

## 📸 Screenshots

### 1. Welcome Screen
![Welcome Screen](Screenshots/1.%20Landing%20or%20Home%20Page.png)
*The main landing page with system overview and getting started instructions*

### 2. Dataset Overview
![Dataset Overview 1](Screenshots/2.%20Dataset%20Overview%201.png)
![Dataset Overview 2](Screenshots/3.%20Dataset%20Overview%202.png)
*Comprehensive view of dataset statistics, sample data, and data quality metrics*

### 3. Basic Visualizations
![Basic Visualizations](Screenshots/4.%20Data%20Visualization.png)
*Detailed Data Visualizations*

### 4. Data Preprocessing
![Data Preprocessing](Screenshots/5.%20Data%20Preprocessing.png)
*Automated data cleaning, outlier handling, and feature engineering process*

### 5. Model Training & Evaluation
![Model Training](Screenshots/6.%20Model%20Training.png)
*Machine learning model performance metrics and evaluation results*

### 6. Model Performance Visualization
![Model Performance](Screenshots/7.%20Model%20Performance%20Visualization.png)
*Comprehensive visualizations of model performance and feature importance*

### 7. At-Risk Student Identification
![At-Risk Students](Screenshots/8.%20At%20Risk%20Students.png)
*Identification and analysis of students who need immediate attention*

### 9. Intervention Recommendations
![Recommendations](Screenshots/9.%20Recommendation%20for%20Students.png)
*Personalized intervention strategies for at-risk students*

### 10. Score Prediction Tool
![Score Prediction](Screenshots/10.%20Single%20Student%20Score%20Prediction.png)
*Interactive tool for predicting individual student performance*

### 11. Final Summary
![Final Summary 1](Screenshots/11.%20Final%20Summary%201.png)
![Final Summary 2](Screenshots/12.%20Final%20Summary%202.png)
*Comprehensive analysis summary and key takeaways*

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/SuccessNEERAJ/Student-Exam-Score-Prediction-Webapp
   ```

2. **Install required packages**
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn scipy
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the application**
   Open your web browser and navigate to `http://localhost:8501`

## 📊 Usage

### Getting Started

1. **Upload Your Dataset**: Use the sidebar to upload a CSV file containing student performance data
2. **Explore Data**: Navigate through different analysis options using the dropdown menu
3. **Preprocess Data**: Clean and prepare your data for machine learning
4. **Train Model**: Build and evaluate the prediction model
5. **Identify At-Risk Students**: Find students who need intervention
6. **Get Recommendations**: Receive personalized intervention strategies
7. **Make Predictions**: Use the prediction tool for individual students

### Navigation Options

- **📊 Data Overview**: Basic dataset information and statistics
- **📈 Data Visualization**: Interactive charts and distributions
- **🔧 Data Preprocessing**: Automated data cleaning and feature engineering
- **🤖 Model Training**: Machine learning model development and evaluation
- **📊 Model Performance**: Detailed performance analysis and visualizations
- **⚠️ At-Risk Students**: Risk assessment and student identification
- **💡 Recommendations**: Personalized intervention strategies
- **🔮 Score Prediction**: Individual student performance prediction
- **📋 Final Summary**: Comprehensive analysis summary

## 📁 Dataset Format

Your CSV file should include the following columns:

### Required Columns
- `Student_ID`: Unique identifier for each student
- `Age`: Student age
- `Gender`: Male/Female
- `Department`: Academic department (CS, Engineering, Business, Mathematics)
- `Attendance (%)`: Attendance percentage
- `Midterm_Score`: Midterm examination score
- `Final_Score`: Final examination score
- `Total_Score`: Overall academic score (target variable)
- `Study_Hours_per_Week`: Weekly study hours
- `Stress_Level (1-10)`: Stress level on a scale of 1-10
- `Sleep_Hours_per_Night`: Average sleep hours per night

### Optional Columns
- `First_Name`, `Last_Name`: Student names
- `Email`: Student email address
- `Assignments_Avg`: Average assignment score
- `Quizzes_Avg`: Average quiz score
- `Participation_Score`: Class participation score
- `Projects_Score`: Project scores
- `Extracurricular_Activities`: Yes/No
- `Internet_Access_at_Home`: Yes/No
- `Parent_Education_Level`: Education level of parents
- `Family_Income_Level`: Low/Medium/High

### Sample Data Format
```csv
Student_ID,Age,Gender,Department,Attendance (%),Midterm_Score,Final_Score,Total_Score,Study_Hours_per_Week,Stress_Level (1-10),Sleep_Hours_per_Night
S1000,22,Female,Mathematics,97.36,40.61,59.61,59.89,10.3,1,5.9
S1001,18,Male,Business,97.71,57.27,74.0,81.92,27.1,4,4.3
S1002,24,Male,Engineering,99.52,41.84,63.85,67.72,12.4,9,6.1
```

## 🔧 Technical Details

### Machine Learning Model
- **Algorithm**: Random Forest Regressor
- **Features**: 15+ student attributes including engineered features
- **Performance**: ~98% R² Score on test data
- **Cross-validation**: 5-fold cross-validation for robust evaluation

### Feature Engineering
The system automatically creates composite scores:
- **Academic Performance Score**: Weighted combination of exam and assignment scores
- **Lifestyle Score**: Sleep, stress, and study habits composite
- **Support System Score**: Family income, parent education, and resources
- **Engagement Score**: Attendance, participation, and extracurricular activities

### Risk Assessment Criteria
Students are evaluated on 8 risk factors:
1. Low Total Score (< 50)
2. High Prediction Error (> 10 points)
3. Low Attendance (< 70%)
4. High Stress Level (≥ 8/10)
5. Insufficient Sleep (< 5 hours)
6. Low Study Hours (< 10 hours/week)
7. Poor Academic Performance (Midterm/Final < 50)
8. Low Participation (< 3/10)

### Technology Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Statistical Analysis**: scipy

## 📁 Project Structure

```
student-performance-prediction/
├── streamlit_app.py              # Main Streamlit application
├── Student_Exam_Score_Prediction.ipynb  # Original google colab script
├── Dataset/
│   └── Students Performance Dataset.csv  # Sample dataset
├── Screenshots/                  # Application screenshots
│   ├── 1. Landing or Home Page.png
│   ├── 2. Dataset Overview 1.png
│   └── ...
└── README.md                    # Project documentation
```

## 🤝 Contributing

We welcome contributions to improve the Student Performance Prediction System! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Contribution
- Additional machine learning models
- Enhanced visualization features
- Mobile-responsive design
- API development
- Documentation improvements
- Bug fixes and optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ If you find this project helpful, please consider giving it a star on GitHub!**

*Built with ❤️ for educational excellence and student success*
