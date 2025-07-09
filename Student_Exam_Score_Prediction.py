
#!/usr/bin/env python3
"""
Student Performance Prediction and Academic Monitoring System
============================================================

This script provides a comprehensive system for predicting student performance
and identifying at-risk students using machine learning techniques.

Author: Academic Analytics Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class StudentPerformancePredictor:
    """
    A comprehensive system for predicting student performance and identifying at-risk students.
    """
    
    def __init__(self, csv_file_path):
        """
        Initialize the predictor with a dataset.
        
        Args:
            csv_file_path (str): Path to the student performance dataset CSV file
        """
        self.csv_file_path = csv_file_path
        self.df = None
        self.df_processed = None
        self.df_final = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoders = {}
        self.best_model = None
        self.results = {}
        self.feature_columns = []
        
        # Set style for better visualizations
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_and_explore_data(self):
        """Load the dataset and perform initial exploration."""
        print("Loading dataset...")
        self.df = pd.read_csv(self.csv_file_path)
        
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Display basic information about the dataset
        print("\n=== DATASET OVERVIEW ===")
        print(f"Number of records: {len(self.df)}")
        print(f"Number of features: {len(self.df.columns)}")
        print("\n=== FIRST 5 ROWS ===")
        print(self.df.head())
        
        print("\n=== DATA TYPES ===")
        print(self.df.dtypes)
        
        print("\n=== BASIC STATISTICS ===")
        print(self.df.describe())
        
        # Check for missing values
        print("\n=== MISSING VALUES ===")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
        
    def visualize_data_distribution(self):
        """Create comprehensive visualization of the dataset."""
        print("Creating data distribution visualizations...")
        
        # Create comprehensive visualization of the dataset
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Student Performance Dataset - Key Variables Distribution', fontsize=16, fontweight='bold')
        
        # Distribution of Total Score (target variable)
        axes[0, 0].hist(self.df['Total_Score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Total Score')
        axes[0, 0].set_xlabel('Total Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # Distribution of Age
        axes[0, 1].hist(self.df['Age'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Distribution of Age')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Frequency')
        
        # Distribution of Attendance
        axes[0, 2].hist(self.df['Attendance (%)'], bins=25, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_title('Distribution of Attendance %')
        axes[0, 2].set_xlabel('Attendance %')
        axes[0, 2].set_ylabel('Frequency')
        
        # Distribution of Study Hours
        axes[0, 3].hist(self.df['Study_Hours_per_Week'], bins=25, alpha=0.7, color='pink', edgecolor='black')
        stress_mean = self.df['Study_Hours_per_Week'].mean()
        axes[0, 3].axvline(stress_mean, color='red', linestyle='--', label=f'Mean: {stress_mean:.2f}')
        axes[0, 3].set_title('Distribution of Study Hours per Week')
        axes[0, 3].set_xlabel('Study Hours per Week')
        axes[0, 3].set_ylabel('Frequency')
        axes[0, 3].legend()
        
        # Gender distribution
        gender_counts = self.df['Gender'].value_counts()
        axes[1, 0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Gender Distribution')
        
        # Department distribution
        dept_counts = self.df['Department'].value_counts()
        axes[1, 1].bar(dept_counts.index, dept_counts.values, color='lightcoral')
        axes[1, 1].set_title('Department Distribution')
        axes[1, 1].set_xlabel('Department')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Grade distribution
        grade_counts = self.df['Grade'].value_counts()
        axes[1, 2].bar(grade_counts.index, grade_counts.values, color='lightblue')
        axes[1, 2].set_title('Grade Distribution')
        axes[1, 2].set_xlabel('Grade')
        axes[1, 2].set_ylabel('Count')
        
        # Family Income Level distribution
        income_counts = self.df['Family_Income_Level'].value_counts()
        axes[1, 3].pie(income_counts.values, labels=income_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1, 3].set_title('Family Income Level Distribution')
        
        # Stress Level distribution
        axes[2, 0].hist(self.df['Stress_Level (1-10)'], bins=10, alpha=0.7, color='red', edgecolor='black')
        axes[2, 0].set_title('Stress Level Distribution')
        axes[2, 0].set_xlabel('Stress Level (1-10)')
        axes[2, 0].set_ylabel('Frequency')
        
        # Sleep Hours distribution
        axes[2, 1].hist(self.df['Sleep_Hours_per_Night'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[2, 1].set_title('Sleep Hours per Night Distribution')
        axes[2, 1].set_xlabel('Sleep Hours per Night')
        axes[2, 1].set_ylabel('Frequency')
        
        # Parent Education Level distribution
        parent_edu_counts = self.df['Parent_Education_Level'].value_counts()
        axes[2, 2].bar(parent_edu_counts.index, parent_edu_counts.values, color='gold')
        axes[2, 2].set_title('Parent Education Level Distribution')
        axes[2, 2].set_xlabel('Parent Education Level')
        axes[2, 2].set_ylabel('Count')
        axes[2, 2].tick_params(axis='x', rotation=45)
        
        # Extracurricular Activities distribution
        extra_counts = self.df['Extracurricular_Activities'].value_counts()
        axes[2, 3].pie(extra_counts.values, labels=extra_counts.index, autopct='%1.1f%%', startangle=90)
        axes[2, 3].set_title('Extracurricular Activities Distribution')
        
        plt.tight_layout()
        plt.show()
        
    def analyze_correlations(self):
        """Analyze correlations between variables."""
        print("Analyzing correlations...")
        
        # Select numeric columns for correlation analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        print("Numeric columns for correlation analysis:")
        print(numeric_cols)
        
        # Create correlation matrix
        correlation_matrix = self.df[numeric_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(14, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                    annot=True, 
                    cmap='coolwarm', 
                    vmin=-1, 
                    vmax=1,
                    center=0,
                    square=True,
                    fmt='.2f',
                    cbar_kws={"shrink": .5},
                    mask=mask)
        plt.title('Correlation Matrix of Numeric Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Show features most correlated with Total_Score
        print("\n=== FEATURES MOST CORRELATED WITH TOTAL SCORE ===")
        total_score_corr = correlation_matrix['Total_Score'].sort_values(ascending=False)
        print(total_score_corr[1:])  # Exclude self-correlation
        
    def preprocess_data(self):
        """Preprocess the data by handling missing values, outliers, and feature engineering."""
        print("Starting data preprocessing...")
        
        # Drop irrelevant columns
        columns_to_drop = [
            'Student_ID',      # Unique identifier, not predictive
            'First_Name',      # Personal information, not predictive
            'Last_Name',       # Personal information, not predictive
            'Email',           # Personal information, not predictive
            'Grade'            # This is derived from Total_Score, would cause data leakage
        ]
        
        print(f"Original dataset shape: {self.df.shape}")
        print(f"Columns to drop: {columns_to_drop}")
        
        # Create a copy for processing
        self.df_processed = self.df.drop(columns=columns_to_drop)
        print(f"Dataset shape after dropping irrelevant columns: {self.df_processed.shape}")
        print(f"Remaining columns: {list(self.df_processed.columns)}")
        
        # Handle missing values
        self._handle_missing_values()
        
        # Handle outliers
        self._handle_outliers()
        
        # Feature engineering
        self._feature_engineering()
        
        # Encode categorical variables
        self._encode_categorical_variables()
        
        print(f"Final dataset shape: {self.df_final.shape}")
        print(f"Final columns: {list(self.df_final.columns)}")
        
    def _handle_missing_values(self):
        """Handle missing values in the dataset."""
        print("=== HANDLING MISSING VALUES ===")
        
        # Check for missing values in the processed dataset
        missing_values = self.df_processed.isnull().sum()
        missing_percentage = (missing_values / len(self.df_processed)) * 100
        missing_info = pd.DataFrame({
            'Missing_Count': missing_values,
            'Missing_Percentage': missing_percentage
        })
        print(missing_info[missing_info['Missing_Count'] > 0])
        
        # For numeric columns, fill with median
        numeric_cols = self.df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df_processed[col].isnull().sum() > 0:
                median_value = self.df_processed[col].median()
                self.df_processed[col].fillna(median_value, inplace=True)
                print(f"Filled missing values in {col} with median: {median_value:.2f}")
        
        # For categorical columns, fill with mode
        categorical_cols = self.df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df_processed[col].isnull().sum() > 0:
                mode_value = self.df_processed[col].mode()[0]
                self.df_processed[col].fillna(mode_value, inplace=True)
                print(f"Filled missing values in {col} with mode: {mode_value}")
        
        # Verify no missing values remain
        print(f"\nMissing values after handling: {self.df_processed.isnull().sum().sum()}")
        
    def _handle_outliers(self):
        """Handle outliers in the dataset."""
        print("=== HANDLING OUTLIERS ===")
        
        # Function to detect outliers using IQR method
        def detect_outliers_iqr(df, columns):
            outliers_info = {}
            
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            
            return outliers_info
        
        # Detect outliers in numeric columns
        numeric_cols = self.df_processed.select_dtypes(include=[np.number]).columns.tolist()
        outliers_info = detect_outliers_iqr(self.df_processed, numeric_cols)
        
        print("Outlier detection results:")
        for col, info in outliers_info.items():
            print(f"{col}: {info['count']} outliers ({info['percentage']:.2f}%)")
        
        # Handle outliers using capping method (less aggressive than removal)
        numeric_cols_for_capping = [col for col in numeric_cols if col != 'Total_Score']
        
        for col in numeric_cols_for_capping:
            Q1 = self.df_processed[col].quantile(0.25)
            Q3 = self.df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            self.df_processed[col] = np.where(self.df_processed[col] < lower_bound, lower_bound, self.df_processed[col])
            self.df_processed[col] = np.where(self.df_processed[col] > upper_bound, upper_bound, self.df_processed[col])
        
        print("Outliers handled using capping method.")
        
    def _feature_engineering(self):
        """Create new features from existing ones."""
        print("=== FEATURE ENGINEERING ===")
        
        # Create a copy for feature engineering
        self.df_final = self.df_processed.copy()
        
        # 1. Academic Performance Score (weighted average of key academic metrics)
        self.df_final['Academic_Performance_Score'] = (
            self.df_final['Midterm_Score'] * 0.3 +
            self.df_final['Final_Score'] * 0.4 +
            self.df_final['Assignments_Avg'] * 0.2 +
            self.df_final['Quizzes_Avg'] * 0.1
        )
        
        # 2. Lifestyle Score (combining sleep, stress, and study hours)
        self.df_final['Lifestyle_Score'] = (
            (10 - self.df_final['Stress_Level (1-10)']) * 0.3 +  # Lower stress is better
            (self.df_final['Sleep_Hours_per_Night'] / 8) * 10 * 0.3 +  # Normalized sleep hours
            (self.df_final['Study_Hours_per_Week'] / 40) * 10 * 0.4  # Normalized study hours
        )
        
        # 3. Support System Score (combining family income, parent education, internet access)
        # First, we need to encode these categorical variables temporarily for calculation
        temp_family_income = LabelEncoder().fit_transform(self.df_final['Family_Income_Level'])
        temp_parent_education = LabelEncoder().fit_transform(self.df_final['Parent_Education_Level'])
        temp_internet = LabelEncoder().fit_transform(self.df_final['Internet_Access_at_Home'])
        
        self.df_final['Support_System_Score'] = (
            temp_family_income * 2 +  # Assuming 0=Low, 1=Medium, 2=High
            temp_parent_education * 1.5 +  # Higher education = better support
            temp_internet * 2  # Internet access is important
        )
        
        # 4. Engagement Score (combining attendance, participation, and extracurricular)
        temp_extracurricular = LabelEncoder().fit_transform(self.df_final['Extracurricular_Activities'])
        
        self.df_final['Engagement_Score'] = (
            (self.df_final['Attendance (%)'] / 100) * 10 * 0.4 +
            self.df_final['Participation_Score'] * 1.0 * 0.3 +
            temp_extracurricular * 3 * 0.3
        )
        
        print("Created new features:")
        print("- Academic_Performance_Score")
        print("- Lifestyle_Score")
        print("- Support_System_Score")
        print("- Engagement_Score")
        
    def _encode_categorical_variables(self):
        """Encode categorical variables."""
        print("=== ENCODING CATEGORICAL VARIABLES ===")
        
        categorical_cols = self.df_final.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df_final[col] = le.fit_transform(self.df_final[col])
            self.label_encoders[col] = le
            print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
            
    def prepare_for_training(self):
        """Prepare data for model training."""
        print("Preparing data for training...")
        
        # Define features and target
        target_column = 'Total_Score'
        self.feature_columns = [col for col in self.df_final.columns if col != target_column]
        
        X = self.df_final[self.feature_columns]
        y = self.df_final[target_column]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Features scaled using StandardScaler.")
        
    def train_random_forest(self):
        """Train and evaluate Random Forest model."""
        print("=== RANDOM FOREST MODEL TRAINING AND EVALUATION ===")
        
        # Initialize Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        print("Training Random Forest...")
        
        # Train the model
        rf_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_train = rf_model.predict(self.X_train)
        y_pred_test = rf_model.predict(self.X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, y_pred_train)
        test_mse = mean_squared_error(self.y_test, y_pred_test)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(rf_model, self.X_train, self.y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Store results
        self.results['Random Forest'] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'model': rf_model
        }
        
        self.best_model = rf_model
        
        print(f"\n=== RANDOM FOREST PERFORMANCE METRICS ===")
        print(f"Training R² Score: {train_r2:.4f}")
        print(f"Test R² Score: {test_r2:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Cross-validation R²: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        
        # Check for overfitting
        print(f"\n=== OVERFITTING ANALYSIS ===")
        r2_diff = train_r2 - test_r2
        print(f"R² Difference (Train - Test): {r2_diff:.4f}")
        if r2_diff > 0.1:
            print("⚠️  Model shows signs of overfitting")
        elif r2_diff < 0.05:
            print("✅ Model shows good generalization")
        else:
            print("🔄 Model shows moderate overfitting")
            
    def visualize_model_performance(self):
        """Visualize model performance."""
        print("Creating model performance visualizations...")
        
        if not self.best_model:
            print("No model trained yet. Please train a model first.")
            return
            
        # Get results for the best model
        best_result = self.results['Random Forest']
        
        # Display performance metrics in a table format
        performance_metrics = pd.DataFrame({
            'Metric': ['Train R²', 'Test R²', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'CV Mean R²', 'CV Std R²'],
            'Value': [best_result['train_r2'], best_result['test_r2'], best_result['train_rmse'], 
                     best_result['test_rmse'], best_result['train_mae'], best_result['test_mae'], 
                     best_result['cv_mean'], best_result['cv_std']]
        })
        
        print("=== RANDOM FOREST PERFORMANCE SUMMARY ===")
        print(performance_metrics.round(4))
        
        # Visualize model performance
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Random Forest Model Performance', fontsize=16, fontweight='bold')
        
        # 1. Training vs Test Performance Comparison
        metrics = ['R²', 'RMSE', 'MAE']
        train_values = [best_result['train_r2'], best_result['train_rmse'], best_result['train_mae']]
        test_values = [best_result['test_r2'], best_result['test_rmse'], best_result['test_mae']]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, train_values, width, label='Train', alpha=0.8, color='lightblue')
        axes[0, 0].bar(x_pos + width/2, test_values, width, label='Test', alpha=0.8, color='lightcoral')
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Values')
        axes[0, 0].set_title('Training vs Test Performance')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(metrics)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Cross-validation scores
        axes[0, 1].bar(['CV R² Score'], [best_result['cv_mean']], yerr=[best_result['cv_std']], 
                       capsize=10, alpha=0.8, color='green')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('Cross-validation R² Score with Standard Deviation')
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # 3. Model complexity visualization (Tree depth distribution)
        tree_depths = [tree.tree_.max_depth for tree in self.best_model.estimators_]
        axes[1, 0].hist(tree_depths, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_xlabel('Tree Depth')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Tree Depths in Random Forest')
        axes[1, 0].axvline(np.mean(tree_depths), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(tree_depths):.1f}')
        axes[1, 0].legend()
        
        plt.tight_layout()
        plt.show()
    
    def analyze_feature_importance(model, feature_columns):
        """
        Analyze and visualize feature importance
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_columns (list): List of feature names
        """
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Top 15 Most Important Features:")
        print(feature_importance.head(15))
        
        # Plot feature importance
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Random Forest Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Top 15 Feature Importance
        top_features = feature_importance.head(15)
        axes[0].barh(top_features['Feature'], top_features['Importance'])
        axes[0].set_xlabel('Importance Score')
        axes[0].set_title('Top 15 Feature Importance')
        axes[0].invert_yaxis()
        
        # 2. Feature Importance Distribution
        axes[1].hist(feature_importance['Importance'], bins=20, alpha=0.7, 
                    color='skyblue', edgecolor='black')
        axes[1].set_xlabel('Importance Score')
        axes[1].set_ylabel('Number of Features')
        axes[1].set_title('Distribution of Feature Importance Scores')
        axes[1].axvline(feature_importance['Importance'].mean(), color='red', 
                        linestyle='--', label=f'Mean: {feature_importance["Importance"].mean():.4f}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Feature importance insights
        print(f"\n=== FEATURE IMPORTANCE INSIGHTS ===")
        print(f"Most important feature: {feature_importance.iloc[0]['Feature']} ({feature_importance.iloc[0]['Importance']:.4f})")
        print(f"Least important feature: {feature_importance.iloc[-1]['Feature']} ({feature_importance.iloc[-1]['Importance']:.4f})")
        print(f"Average importance: {feature_importance['Importance'].mean():.4f}")
        
        # Categorize features by importance
        high_importance = feature_importance[feature_importance['Importance'] > 0.05]
        medium_importance = feature_importance[(feature_importance['Importance'] >= 0.02) & 
                                            (feature_importance['Importance'] <= 0.05)]
        low_importance = feature_importance[feature_importance['Importance'] < 0.02]
        
        print(f"\nFeature Categories:")
        print(f"High importance (>5%): {len(high_importance)} features")
        print(f"Medium importance (2-5%): {len(medium_importance)} features")
        print(f"Low importance (<2%): {len(low_importance)} features")
        
        return feature_importance

    def plot_model_predictions(model, X_test, y_test, model_name):
        """
        Create residual plots and prediction analysis
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
        """
        print(f"\n=== PREDICTION ANALYSIS - {model_name} ===")
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Create residual plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - Prediction Analysis', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, predictions, alpha=0.6, color='blue')
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Total Score')
        axes[0, 0].set_ylabel('Predicted Total Score')
        axes[0, 0].set_title('Actual vs Predicted (Test Set)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation, _ = pearsonr(y_test, predictions)
        axes[0, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=axes[0, 0].transAxes, fontsize=12, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Residuals vs Predicted
        residuals = y_test - predictions
        axes[0, 1].scatter(predictions, residuals, alpha=0.6, color='red')
        axes[0, 1].axhline(y=0, color='black', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Total Score')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].axvline(residuals.mean(), color='orange', linestyle='--', alpha=0.7, 
                        label=f'Mean: {residuals.mean():.2f}')
        axes[1, 0].legend()
        
        # 4. Q-Q plot of residuals
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate and display residual statistics
        print(f"\n=== RESIDUAL ANALYSIS ===")
        print(f"Mean of residuals: {np.mean(residuals):.4f}")
        print(f"Standard deviation of residuals: {np.std(residuals):.4f}")
        print(f"Median of residuals: {np.median(residuals):.4f}")
        print(f"Min residual: {np.min(residuals):.4f}")
        print(f"Max residual: {np.max(residuals):.4f}")
        
        # Prediction accuracy analysis
        within_5_points = np.sum(np.abs(residuals) <= 5)
        within_10_points = np.sum(np.abs(residuals) <= 10)
        within_15_points = np.sum(np.abs(residuals) <= 15)
        
        print(f"\n=== PREDICTION ACCURACY ===")
        print(f"Predictions within 5 points: {within_5_points}/{len(residuals)} ({(within_5_points/len(residuals))*100:.1f}%)")
        print(f"Predictions within 10 points: {within_10_points}/{len(residuals)} ({(within_10_points/len(residuals))*100:.1f}%)")
        print(f"Predictions within 15 points: {within_15_points}/{len(residuals)} ({(within_15_points/len(residuals))*100:.1f}%)")

    def identify_at_risk_students(df, model, X):
        """
        Identify students who need attention based on multiple risk factors
        
        Args:
            df (pd.DataFrame): Original dataframe
            model: Trained model
            X (np.array): Feature matrix
            
        Returns:
            tuple: (at_risk_summary, high_priority_students, risk_scores)
        """
        print("\n=== IDENTIFYING AT-RISK STUDENTS ===")
        
        # Make predictions for the entire dataset
        all_predictions = model.predict(X)
        
        # Add predictions to the original dataset for analysis
        df_analysis = df.copy()
        df_analysis['Predicted_Score'] = all_predictions
        df_analysis['Prediction_Error'] = df_analysis['Total_Score'] - all_predictions
        
        # Define criteria for students needing attention
        criteria = {
            'Low Total Score': df_analysis['Total_Score'] < 50,
            'High Prediction Error': abs(df_analysis['Prediction_Error']) > 10,
            'Low Attendance': df_analysis['Attendance (%)'] < 70,
            'High Stress': df_analysis['Stress_Level (1-10)'] >= 8,
            'Low Sleep': df_analysis['Sleep_Hours_per_Night'] < 5,
            'Low Study Hours': df_analysis['Study_Hours_per_Week'] < 10,
            'Poor Academic Performance': (df_analysis['Midterm_Score'] < 50) | (df_analysis['Final_Score'] < 50),
            'Low Participation': df_analysis['Participation_Score'] < 3
        }
        
        # Calculate risk scores for each student
        risk_scores = pd.DataFrame(index=df_analysis.index)
        
        for criterion, condition in criteria.items():
            risk_scores[criterion] = condition.astype(int)
        
        # Count total risk factors per student
        risk_scores['Total_Risk_Factors'] = risk_scores.sum(axis=1)
        criteria['Multiple Risk Factors'] = risk_scores['Total_Risk_Factors'] >= 3
        
        # Create summary of at-risk students
        at_risk_summary = pd.DataFrame({
            'Criterion': list(criteria.keys()),
            'Number_of_Students': [criteria[criterion].sum() for criterion in criteria.keys()],
            'Percentage': [(criteria[criterion].sum() / len(df_analysis)) * 100 for criterion in criteria.keys()]
        })
        
        print("=== AT-RISK STUDENTS SUMMARY ===")
        print(at_risk_summary.round(2))
        
        # Students with multiple risk factors (highest priority)
        high_priority_students = df_analysis[criteria['Multiple Risk Factors']].copy()
        high_priority_students['Risk_Score'] = risk_scores['Total_Risk_Factors'][criteria['Multiple Risk Factors']]
        
        print(f"\n=== HIGH PRIORITY STUDENTS (Multiple Risk Factors) ===")
        print(f"Number of high priority students: {len(high_priority_students)}")
        print(f"Percentage of total students: {(len(high_priority_students)/len(df_analysis))*100:.2f}%")
        
        # Display top 10 students needing most attention
        priority_columns = ['Student_ID', 'First_Name', 'Last_Name', 'Department', 'Total_Score', 
                        'Predicted_Score', 'Attendance (%)', 'Stress_Level (1-10)', 
                        'Sleep_Hours_per_Night', 'Study_Hours_per_Week', 'Risk_Score']
        
        top_priority = high_priority_students.nlargest(10, 'Risk_Score')[priority_columns]
        print("\nTop 10 students needing immediate attention:")
        print(top_priority.to_string(index=False))
        
        return at_risk_summary, high_priority_students, risk_scores, df_analysis

    def analyze_risk_patterns(df_analysis, risk_scores):
        """
        Analyze patterns in at-risk students across different demographics
        
        Args:
            df_analysis (pd.DataFrame): Analysis dataframe
            risk_scores (pd.DataFrame): Risk scores dataframe
        """
        print("\n=== DETAILED ANALYSIS OF AT-RISK STUDENTS ===")
        
        # Multiple risk factors condition
        multiple_risk_condition = risk_scores['Total_Risk_Factors'] >= 3
        
        # Department-wise analysis
        dept_risk_counts = pd.DataFrame({
            'Department': df_analysis['Department'].unique(),
            'At_Risk_Count': [multiple_risk_condition[df_analysis['Department'] == dept].sum() 
                            for dept in df_analysis['Department'].unique()],
            'Total_Students': [len(df_analysis[df_analysis['Department'] == dept]) 
                            for dept in df_analysis['Department'].unique()]
        })
        dept_risk_counts['At_Risk_Percentage'] = (dept_risk_counts['At_Risk_Count'] / dept_risk_counts['Total_Students']) * 100
        
        print("\nDepartment-wise Risk Analysis:")
        print(dept_risk_counts.round(2))
        
        # Gender-wise analysis
        gender_risk = pd.DataFrame({
            'Gender': df_analysis['Gender'].unique(),
            'At_Risk_Count': [multiple_risk_condition[df_analysis['Gender'] == gender].sum() 
                            for gender in df_analysis['Gender'].unique()],
            'Total_Students': [len(df_analysis[df_analysis['Gender'] == gender]) 
                            for gender in df_analysis['Gender'].unique()]
        })
        gender_risk['At_Risk_Percentage'] = (gender_risk['At_Risk_Count'] / gender_risk['Total_Students']) * 100
        
        print("\nGender-wise Risk Analysis:")
        print(gender_risk.round(2))
        
        # Age group analysis
        df_analysis['Age_Group'] = pd.cut(df_analysis['Age'], bins=[17, 20, 22, 25, 30], 
                                        labels=['18-20', '21-22', '23-25', '26+'])
        age_risk = df_analysis.groupby('Age_Group').agg({
            'Total_Score': 'mean',
            'Stress_Level (1-10)': 'mean',
            'Sleep_Hours_per_Night': 'mean'
        }).round(2)
        
        print("\nAge Group Analysis:")
        print(age_risk)
        
        return dept_risk_counts, gender_risk

    def visualize_risk_analysis(at_risk_summary, dept_risk_counts, gender_risk, risk_scores, df_analysis):
        """
        Create comprehensive visualization of at-risk students
        
        Args:
            at_risk_summary (pd.DataFrame): Summary of risk factors
            dept_risk_counts (pd.DataFrame): Department-wise risk counts
            gender_risk (pd.DataFrame): Gender-wise risk analysis
            risk_scores (pd.DataFrame): Risk scores dataframe
            df_analysis (pd.DataFrame): Analysis dataframe
        """
        print("\n=== VISUALIZING AT-RISK STUDENTS ANALYSIS ===")
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('At-Risk Students Analysis', fontsize=16, fontweight='bold')
        
        # 1. Risk factors distribution
        axes[0, 0].bar(at_risk_summary['Criterion'], at_risk_summary['Number_of_Students'], 
                    color='lightcoral', alpha=0.7)
        axes[0, 0].set_title('Distribution of Risk Factors')
        axes[0, 0].set_xlabel('Risk Criterion')
        axes[0, 0].set_ylabel('Number of Students')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Department-wise at-risk percentage
        axes[0, 1].bar(dept_risk_counts['Department'], dept_risk_counts['At_Risk_Percentage'], 
                    color='orange', alpha=0.7)
        axes[0, 1].set_title('At-Risk Students by Department')
        axes[0, 1].set_xlabel('Department')
        axes[0, 1].set_ylabel('At-Risk Percentage (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Risk score distribution
        axes[0, 2].hist(risk_scores['Total_Risk_Factors'], bins=8, alpha=0.7, 
                        color='red', edgecolor='black')
        axes[0, 2].set_title('Distribution of Risk Scores')
        axes[0, 2].set_xlabel('Number of Risk Factors')
        axes[0, 2].set_ylabel('Number of Students')
        
        # 4. Total Score vs Risk Score
        axes[1, 0].scatter(risk_scores['Total_Risk_Factors'], df_analysis['Total_Score'], 
                        alpha=0.6, color='purple')
        axes[1, 0].set_title('Total Score vs Risk Score')
        axes[1, 0].set_xlabel('Number of Risk Factors')
        axes[1, 0].set_ylabel('Total Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Stress vs Sleep Hours (colored by risk level)
        high_risk_mask = risk_scores['Total_Risk_Factors'] >= 3
        axes[1, 1].scatter(df_analysis[~high_risk_mask]['Stress_Level (1-10)'], 
                        df_analysis[~high_risk_mask]['Sleep_Hours_per_Night'], 
                        alpha=0.6, color='green', label='Low Risk')
        axes[1, 1].scatter(df_analysis[high_risk_mask]['Stress_Level (1-10)'], 
                        df_analysis[high_risk_mask]['Sleep_Hours_per_Night'], 
                        alpha=0.6, color='red', label='High Risk')
        axes[1, 1].set_title('Stress vs Sleep Hours')
        axes[1, 1].set_xlabel('Stress Level (1-10)')
        axes[1, 1].set_ylabel('Sleep Hours per Night')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Gender-wise at-risk percentage
        axes[1, 2].bar(gender_risk['Gender'], gender_risk['At_Risk_Percentage'], 
                    color='lightblue', alpha=0.7)
        axes[1, 2].set_title('At-Risk Students by Gender')
        axes[1, 2].set_xlabel('Gender')
        axes[1, 2].set_ylabel('At-Risk Percentage (%)')
        
        plt.tight_layout()
        plt.show()

    def generate_recommendations(student_data):
        """
        Generate personalized recommendations for at-risk students
        
        Args:
            student_data (pd.Series): Student data
            
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        # Academic performance recommendations
        if student_data['Total_Score'] < 50:
            recommendations.append("📚 ACADEMIC SUPPORT: Immediate tutoring and academic counseling required")
        
        if student_data['Attendance (%)'] < 70:
            recommendations.append("🎯 ATTENDANCE: Implement attendance monitoring and intervention program")
        
        if student_data['Midterm_Score'] < 50 or student_data['Final_Score'] < 50:
            recommendations.append("📖 EXAM PREPARATION: Provide exam strategies and study skills workshops")
        
        if student_data['Participation_Score'] < 3:
            recommendations.append("🗣️ ENGAGEMENT: Encourage class participation through interactive activities")
        
        # Lifestyle recommendations
        if student_data['Stress_Level (1-10)'] >= 8:
            recommendations.append("🧘 STRESS MANAGEMENT: Refer to counseling services and stress reduction programs")
        
        if student_data['Sleep_Hours_per_Night'] < 5:
            recommendations.append("😴 SLEEP HYGIENE: Educate about importance of adequate sleep (7-9 hours)")
        
        if student_data['Study_Hours_per_Week'] < 10:
            recommendations.append("⏰ STUDY SCHEDULE: Help create structured study timetable")
        
        # Support system recommendations
        if 'Internet_Access_at_Home' in student_data and student_data['Internet_Access_at_Home'] == 'No':
            recommendations.append("🌐 DIGITAL ACCESS: Provide access to computer labs and internet facilities")
        
        if 'Family_Income_Level' in student_data and student_data['Family_Income_Level'] == 'Low':
            recommendations.append("💰 FINANCIAL SUPPORT: Connect with financial aid and scholarship programs")
        
        if 'Extracurricular_Activities' in student_data and student_data['Extracurricular_Activities'] == 'No':
            recommendations.append("🎭 ENGAGEMENT: Encourage participation in clubs and extracurricular activities")
        
        return recommendations

    def provide_intervention_recommendations(high_priority_students, df_analysis):
        """
        Provide personalized intervention recommendations for high-priority students
        
        Args:
            high_priority_students (pd.DataFrame): High priority students dataframe
            df_analysis (pd.DataFrame): Analysis dataframe
        """
        print("\n=== PERSONALIZED INTERVENTION RECOMMENDATIONS ===")
        
        # Get top 10 priority students
        top_priority = high_priority_students.nlargest(10, 'Risk_Score')
        
        for idx, student in top_priority.iterrows():
            print(f"\n🎓 STUDENT: {student['First_Name']} {student['Last_Name']} (ID: {student['Student_ID']})")
            print(f"   Department: {student['Department']}")
            print(f"   Current Score: {student['Total_Score']:.1f} | Predicted: {student['Predicted_Score']:.1f}")
            print(f"   Risk Score: {student['Risk_Score']}/8")
            
            # Get original data for this student
            original_data = df_analysis.loc[idx]
            recommendations = generate_recommendations(original_data)
            
            print("   📋 RECOMMENDED INTERVENTIONS:")
            for rec in recommendations:
                print(f"      • {rec}")
            
            if len(recommendations) == 0:
                print("      • ✅ Student shows good overall performance with minimal risk factors")

    def generate_final_summary(df_analysis, feature_columns, best_model_name, results, high_priority_students, at_risk_summary):
        """
        Generate comprehensive final summary of the analysis
        
        Args:
            df_analysis (pd.DataFrame): Analysis dataframe
            feature_columns (list): List of feature columns
            best_model_name (str): Name of best performing model
            results (dict): Model results dictionary
            high_priority_students (pd.DataFrame): High priority students
            at_risk_summary (pd.DataFrame): At-risk summary
        """
        print("\n" + "="*80)
        print("FINAL PROJECT SUMMARY")
        print("="*80)
        
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   • Total students analyzed: {len(df_analysis):,}")
        print(f"   • Features used for prediction: {len(feature_columns)}")
        print(f"   • Target variable: Total Score (0-100)")
        
        print(f"\n🔍 DATA QUALITY:")
        print(f"   • Missing values handled: ✅")
        print(f"   • Outliers processed: ✅")
        print(f"   • Feature engineering applied: ✅")
