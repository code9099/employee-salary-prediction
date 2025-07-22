import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💰",
    layout="wide"
)

# Generate sample dataset
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    age = np.random.randint(22, 65, n_samples)
    experience = np.random.randint(0, 40, n_samples)
    education_levels = ['Bachelor', 'Master', 'PhD']
    education = np.random.choice(education_levels, n_samples)
    job_titles = ['Software Engineer', 'Data Scientist', 'Manager', 'Senior Engineer', 'Lead Developer']
    job_title = np.random.choice(job_titles, n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples)
    
    # Create realistic salary based on factors
    base_salary = 40000
    experience_bonus = experience * 2000
    education_bonus = {'Bachelor': 0, 'Master': 15000, 'PhD': 30000}
    job_bonus = {
        'Software Engineer': 5000,
        'Data Scientist': 20000,
        'Manager': 25000,
        'Senior Engineer': 30000,
        'Lead Developer': 35000
    }
    
    salary = (base_salary + 
             experience_bonus + 
             [education_bonus[ed] for ed in education] +
             [job_bonus[job] for job in job_title] +
             np.random.normal(0, 5000, n_samples))
    
    # Ensure positive salaries
    salary = np.maximum(salary, 30000)
    
    df = pd.DataFrame({
        'Age': age,
        'Experience': experience,
        'Education': education,
        'Job_Title': job_title,
        'Gender': gender,
        'Salary': salary
    })
    
    return df

# Load and preprocess data
@st.cache_data
def preprocess_data():
    df = generate_sample_data()
    
    # Encode categorical variables
    le_education = LabelEncoder()
    le_job = LabelEncoder()
    le_gender = LabelEncoder()
    
    df['Education_Encoded'] = le_education.fit_transform(df['Education'])
    df['Job_Title_Encoded'] = le_job.fit_transform(df['Job_Title'])
    df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
    
    return df, le_education, le_job, le_gender

# Train models
@st.cache_data
def train_models():
    df, le_education, le_job, le_gender = preprocess_data()
    
    # Prepare features and target
    features = ['Age', 'Experience', 'Education_Encoded', 'Job_Title_Encoded', 'Gender_Encoded']
    X = df[features]
    y = df['Salary']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'R² Score': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred)
        }
        trained_models[name] = model
    
    return trained_models, results, le_education, le_job, le_gender, X_test, y_test

# Main application
def main():
    st.title("💰 Employee Salary Prediction System")
    st.markdown("### Predict employee salaries using machine learning algorithms")
    
    # Load models and data
    trained_models, results, le_education, le_job, le_gender, X_test, y_test = train_models()
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["🔮 Salary Prediction", "📊 Model Performance", "📈 Data Analysis"])
    
    if page == "🔮 Salary Prediction":
        prediction_page(trained_models, le_education, le_job, le_gender)
    elif page == "📊 Model Performance":
        model_performance_page(results, trained_models, X_test, y_test)
    else:
        data_analysis_page()

def prediction_page(trained_models, le_education, le_job, le_gender):
    st.header("🔮 Salary Prediction")
    st.markdown("Enter employee details to predict salary:")
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("👤 Age", min_value=18, max_value=65, value=28)
        experience = st.number_input("💼 Years of Experience", min_value=0, max_value=40, value=3)
        education = st.selectbox("🎓 Education Level", ["Bachelor", "Master", "PhD"])
    
    with col2:
        job_title = st.selectbox("💻 Job Title", 
                               ["Software Engineer", "Data Scientist", "Manager", 
                                "Senior Engineer", "Lead Developer"])
        gender = st.selectbox("👥 Gender", ["Male", "Female"])
    
    if st.button("💡 Predict Salary", type="primary"):
        # Encode inputs
        education_encoded = le_education.transform([education])[0]
        job_encoded = le_job.transform([job_title])[0]
        gender_encoded = le_gender.transform([gender])[0]
        
        # Prepare input for prediction
        input_data = np.array([[age, experience, education_encoded, job_encoded, gender_encoded]])
        
        # Get predictions from all models
        st.subheader("📊 Predictions from Different Models:")
        
        predictions = {}
        for name, model in trained_models.items():
            pred = model.predict(input_data)[0]
            predictions[name] = pred
            
            if name == "Random Forest":
                st.success(f"🌟 **{name}**: ${pred:,.2f} (Recommended)")
            else:
                st.info(f"📈 **{name}**: ${pred:,.2f}")
        
        # Average prediction
        avg_prediction = np.mean(list(predictions.values()))
        st.markdown("---")
        st.success(f"🎯 **Average Prediction**: ${avg_prediction:,.2f}")
        
        # Additional insights
        st.markdown("### 💡 Insights:")
        st.write(f"• Based on {experience} years of experience")
        st.write(f"• {education} degree holder")
        st.write(f"• Working as {job_title}")

def model_performance_page(results, trained_models, X_test, y_test):
    st.header("📊 Model Performance Analysis")
    
    # Create performance comparison table
    performance_df = pd.DataFrame(results).T
    performance_df = performance_df.round(4)
    
    st.subheader("📈 Model Comparison")
    st.dataframe(performance_df, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 R² Score Comparison")
        r2_scores = [results[model]['R² Score'] for model in results.keys()]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(results.keys(), r2_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_ylabel('R² Score')
        ax.set_title('Model R² Score Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("📉 RMSE Comparison")
        rmse_scores = [results[model]['RMSE'] for model in results.keys()]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(results.keys(), rmse_scores, color=['#FFD93D', '#6BCF7F', '#FF8E53'])
        ax.set_ylabel('RMSE')
        ax.set_title('Model RMSE Comparison')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:,.0f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Feature importance for Random Forest
    st.subheader("🌟 Feature Importance (Random Forest)")
    rf_model = trained_models['Random Forest']
    feature_names = ['Age', 'Experience', 'Education', 'Job Title', 'Gender']
    importance = rf_model.feature_importances_
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feature_names, importance, color='skyblue')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance Analysis')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{width:.3f}', ha='left', va='center')
    
    st.pyplot(fig)

def data_analysis_page():
    st.header("📈 Exploratory Data Analysis")
    
    df, _, _, _ = preprocess_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Salary Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['Salary'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax.set_xlabel('Salary ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Employee Salaries')
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Salary by Job Title")
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column='Salary', by='Job_Title', ax=ax)
        ax.set_title('Salary Distribution by Job Title')
        ax.set_xlabel('Job Title')
        ax.set_ylabel('Salary ($)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("🔗 Feature Correlation Matrix")
    numeric_cols = ['Age', 'Experience', 'Education_Encoded', 'Job_Title_Encoded', 'Gender_Encoded', 'Salary']
    correlation_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    st.pyplot(fig)
    
    # Summary statistics
    st.subheader("📋 Dataset Summary")
    st.dataframe(df.describe(), use_container_width=True)

if __name__ == "__main__":
    main()
