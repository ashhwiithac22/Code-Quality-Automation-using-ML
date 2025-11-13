import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("🚀 ULTIMATE CODE QUALITY AUTOMATION WITH DETAILED METRICS")
print("=" * 70)

dataset_path = "D:\\PROJECTS\\Quality Automation Using ML"
quality_attributes_path = os.path.join(dataset_path, "Software code quality and source code metrics dataset", "quality_attributes")

# Load all detailed metrics
print("📊 LOADING DETAILED CODE METRICS...")
all_detailed_data = []

for root, dirs, files in os.walk(quality_attributes_path):
    for file in files:
        if file.endswith('.csv'):
            full_path = os.path.join(root, file)
            try:
                df = pd.read_csv(full_path)
                
                # Extract project and version
                file_parts = full_path.split('\\')
                project_name = file_parts[-3]  # Project folder
                version = file_parts[-1].replace('.csv', '')  # File name
                
                df['Project'] = project_name
                df['Version'] = version
                all_detailed_data.append(df)
                
            except Exception as e:
                print(f"❌ Error reading {file}: {e}")

# Combine all data
combined_df = pd.concat(all_detailed_data, ignore_index=True)
print(f"✅ LOADED: {combined_df.shape[0]:,} code entities with {combined_df.shape[1]} metrics")

# Data cleaning and preprocessing
print("\n🔧 CLEANING AND PREPROCESSING DATA...")
print("-" * 40)

def clean_detailed_data(df):
    df_clean = df.copy()
    
    # Remove empty rows and useless columns
    df_clean = df_clean.dropna(how='all')
    
    # Remove identifier columns that cause data leakage
    columns_to_drop = ['QualifiedName', 'Name', 'Project', 'Version']
    df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns])
    
    # Convert categorical quality metrics to numerical
    quality_mapping = {'very low': 0, 'low': 1, 'medium': 2, 'high': 3, 'very high': 4, 'extreme': 5}
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                df_clean[col] = df_clean[col].map(quality_mapping).fillna(df_clean[col])
            except:
                # If mapping fails, try to convert to numeric
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Convert all to numeric
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove columns with too many missing values
    missing_threshold = 0.7
    cols_to_keep = df_clean.columns[df_clean.isnull().mean() < missing_threshold]
    df_clean = df_clean[cols_to_keep]
    
    # Fill remaining missing values
    df_clean = df_clean.fillna(df_clean.median())
    
    # Remove constant columns
    df_clean = df_clean.loc[:, df_clean.nunique() > 1]
    
    print(f"   Final shape: {df_clean.shape}")
    print(f"   Remaining features: {len(df_clean.columns)}")
    
    return df_clean

cleaned_detailed = clean_detailed_data(combined_df)

# Create meaningful target variable
print("\n🎯 CREATING CODE QUALITY TARGET VARIABLE...")
print("-" * 40)

# Use complexity-related metrics to define "problematic code"
# High complexity + high coupling + low cohesion = problematic code
complexity_score = (
    cleaned_detailed['Complexity'].fillna(0) + 
    cleaned_detailed['Coupling'].fillna(0) + 
    cleaned_detailed['CBO'].fillna(0) -
    cleaned_detailed['Lack of Cohesion'].fillna(0)  # Lower cohesion is worse
)

# Create binary target: top 20% most problematic code
threshold = complexity_score.quantile(0.8)
y = (complexity_score > threshold).astype(int)

print(f"Target distribution:")
print(y.value_counts())
print(f"Problematic code ratio: {y.mean():.2%}")

# Prepare features
X = cleaned_detailed

print(f"\n📊 FINAL DATASET FOR ML:")
print(f"   Samples: {X.shape[0]:,}")
print(f"   Features: {X.shape[1]}")
print(f"   Target: Binary (problematic vs clean code)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n🤖 TRAINING MACHINE LEARNING MODEL...")
print("-" * 40)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

# Evaluate with proper cross-validation
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print(f"📈 Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Test performance
y_pred = rf_model.predict(X_test_scaled)
test_accuracy = rf_model.score(X_test_scaled, y_test)

print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🎯 TOP 10 FEATURES PREDICTING CODE QUALITY:")
for i, row in feature_importance.head(10).iterrows():
    print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.4f}")

# Save the model
joblib.dump(rf_model, 'ultimate_code_quality_model.pkl')
joblib.dump(scaler, 'ultimate_scaler.pkl')

print(f"\n💾 MODEL SAVED: ultimate_code_quality_model.pkl")

# Create visualization
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(10)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Feature Importance')
plt.title('Top 10 Features for Code Quality Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_detailed.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ ULTIMATE CODE QUALITY AUTOMATION SYSTEM READY!")
print("=" * 70)
print(f"📊 Dataset: {X.shape[0]:,} code entities")
print(f"🎯 Target: Predicting problematic code patterns")
print(f"🤖 Performance: {test_accuracy:.2%} accuracy")
print(f"🔧 Features: {X.shape[1]} code quality metrics")
print(f"💡 Use case: Identify code smells and quality issues automatically")