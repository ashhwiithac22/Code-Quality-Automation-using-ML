import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("🔍 PROPER VALIDATION - CHECKING FOR OVERFITTING")
print("=" * 60)

# Load and inspect data properly
dataset_path = "D:\\PROJECTS\\Quality Automation Using ML"
main_project_path = os.path.join(dataset_path, "Software code quality and source code metrics dataset")
versions_df = pd.read_csv(os.path.join(main_project_path, "versions.csv"))

print("📊 DATA QUALITY CHECK:")
print("-" * 40)

# 1. Check for NULL values
print("1. NULL Values:")
print(versions_df.isnull().sum())

# 2. Check for duplicates
print(f"\n2. Duplicate rows: {versions_df.duplicated().sum()}")

# 3. Check data types
print("\n3. Data Types:")
print(versions_df.dtypes)

# 4. Check target distribution
print("\n4. Target Distribution:")
if 'Number of problematic classes' in versions_df.columns:
    problematic_counts = (versions_df['Number of problematic classes'] > 0).value_counts()
    print(f"Has problematic classes: {problematic_counts}")

# Proper preprocessing
def robust_preprocessing(df):
    df_clean = df.copy()
    
    print("\n🧹 DATA CLEANING:")
    print("-" * 30)
    
    # Remove completely empty rows
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(how='all')
    print(f"Removed {initial_rows - len(df_clean)} completely empty rows")
    
    # Handle numeric columns
    numeric_columns = ['  Commits  ', 'Lines of code', 'Number of classes', 'Number of packages', 
                      'Number of external packages', 'Number of external classes', 
                      'Number of problematic classes', 'Number of highly problematic classes']
    
    for col in numeric_columns:
        if col in df_clean.columns:
            # Clean and convert to numeric
            df_clean[col] = pd.to_numeric(
                df_clean[col].astype(str).str.replace(',', '').str.replace(' ', ''), 
                errors='coerce'
            )
    
    # Remove rows with too many missing values
    missing_threshold = 0.5  # Remove rows with >50% missing values
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(thresh=int(len(df_clean.columns) * (1 - missing_threshold)))
    print(f"Removed {initial_rows - len(df_clean)} rows with too many missing values")
    
    # Fill remaining missing values with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    
    # Feature engineering
    df_clean['problematic_ratio'] = df_clean['Number of problematic classes'] / (df_clean['Number of classes'] + 1)
    df_clean['code_complexity'] = df_clean['Lines of code'] / (df_clean['Number of classes'] + 1)
    
    # Encode repository
    le = LabelEncoder()
    df_clean['Repository_encoded'] = le.fit_transform(df_clean['Repository name'].fillna('Unknown'))
    
    # Create target
    df_clean['has_problematic_classes'] = (df_clean['Number of problematic classes'] > 0).astype(int)
    
    print(f"Final dataset shape: {df_clean.shape}")
    return df_clean

# Apply robust preprocessing
cleaned_df = robust_preprocessing(versions_df)

# Check if we have enough data
if len(cleaned_df) < 20:
    print("❌ WARNING: Too little data after cleaning! Model will likely overfit.")
    
print(f"\n📈 FINAL DATA SHAPE: {cleaned_df.shape}")

# Define features and target
feature_columns = [col for col in ['  Commits  ', 'Lines of code', 'Number of classes', 'Number of packages',
                                  'Number of external packages', 'Number of external classes', 'Repository_encoded',
                                  'problematic_ratio', 'code_complexity'] if col in cleaned_df.columns]

X = cleaned_df[feature_columns]
y = cleaned_df['has_problematic_classes']

print(f"Features: {len(feature_columns)}, Samples: {len(X)}")
print(f"Target distribution:\n{y.value_counts()}")

# 🚨 CRITICAL: Check for data leakage
print("\n🚨 CHECKING FOR DATA LEAKAGE:")
# Check if problematic_ratio (which uses target) is in features
if 'problematic_ratio' in feature_columns:
    print("❌ DATA LEAKAGE DETECTED: 'problematic_ratio' uses target variable in features!")
    print("   This is causing perfect accuracy - REMOVE THIS FEATURE!")
    feature_columns = [col for col in feature_columns if col != 'problematic_ratio']
    X = cleaned_df[feature_columns]
    print("   ✅ Fixed: Removed problematic_ratio from features")

# Proper model validation
print("\n🤖 PROPER MODEL VALIDATION:")
print("-" * 40)

# Use cross-validation instead of simple train-test split
X_clean = X.fillna(X.median())
y_clean = y

# Cross-validation
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)  # Limit depth to prevent overfitting
cv_scores = cross_val_score(rf, X_clean, y_clean, cv=5, scoring='accuracy')

print(f"📊 Cross-Validation Scores: {cv_scores}")
print(f"📈 Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# If cross-validation shows high variance, we have overfitting
if cv_scores.std() > 0.1:
    print("❌ HIGH VARIANCE: Model is overfitting!")

# Now do proper train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean
)

print(f"\n📋 Proper Train-Test Split:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train and evaluate
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 REAL Test Accuracy: {test_accuracy:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n📊 Feature Importance (without data leakage):")
for i, row in feature_importance.iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Confusion Matrix
print(f"\n📋 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n✅ PROPER VALIDATION COMPLETED!")
print("   Now you have realistic performance metrics")