import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("🚀 PROPER DYNAMIC ANALYSIS - NO HARDCODED VALUES")
print("=" * 70)

dataset_path = "D:\\PROJECTS\\Quality Automation Using ML"
quality_attributes_path = os.path.join(dataset_path, "Software code quality and source code metrics dataset", "quality_attributes")

# Step 1: Dynamically load and analyze all projects
print("📊 DYNAMICALLY ANALYZING ALL PROJECTS...")
print("-" * 50)

def safe_ml_analysis(df, project_name):
    """Safely perform ML analysis with proper error handling"""
    try:
        df_clean = df.copy()
        
        # Remove identifier columns
        id_cols = ['QualifiedName', 'Name', 'Project', 'Version', 'File_Source']
        df_clean = df_clean.drop(columns=[col for col in id_cols if col in df_clean.columns])
        
        # Convert categorical to numerical
        quality_map = {'very low': 1, 'low': 2, 'medium': 3, 'high': 4, 'very high': 5}
        
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                try:
                    df_clean[col] = df_clean[col].map(quality_map).fillna(df_clean[col])
                except:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Convert to numeric and handle missing values
        for col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        df_clean = df_clean.fillna(df_clean.median())
        df_clean = df_clean.loc[:, df_clean.nunique() > 1]
        
        # Check for data leakage by ensuring no perfect correlation between features and potential targets
        correlation_matrix = df_clean.corr().abs()
        
        # Find suitable target
        target_candidates = []
        for col in df_clean.columns:
            if df_clean[col].nunique() > 10:
                # Check if this column is not perfectly correlated with any other column
                max_corr = correlation_matrix[col].nlargest(2).iloc[1]  # Second highest (first is self)
                if max_corr < 0.95:  # Not perfectly correlated with another feature
                    threshold = df_clean[col].quantile(0.7)
                    y_temp = (df_clean[col] > threshold).astype(int)
                    balance_ratio = min(y_temp.mean(), 1 - y_temp.mean())
                    
                    if balance_ratio > 0.15:
                        target_candidates.append((col, balance_ratio, max_corr))
        
        if target_candidates:
            # Choose target with best balance and lowest correlation to other features
            target_col, balance_ratio, max_corr = max(target_candidates, key=lambda x: (x[1], -x[2]))
            
            y = (df_clean[target_col] > df_clean[target_col].quantile(0.7)).astype(int)
            feature_cols = [col for col in df_clean.columns if col != target_col]
            X = df_clean[feature_cols]
            
            if len(X) > 100 and len(feature_cols) >= 3:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = RandomForestClassifier(
                    n_estimators=50, max_depth=5, random_state=42, 
                    class_weight='balanced', min_samples_split=50
                )
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                accuracy = (y_pred == y_test).mean()
                
                # Check for suspicious accuracy
                if accuracy > 0.98:
                    return {
                        'accuracy': accuracy,
                        'auc': roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]),
                        'samples': len(X),
                        'target': target_col,
                        'top_feature': feature_cols[np.argmax(model.feature_importances_)],
                        'top_importance': np.max(model.feature_importances_),
                        'issue': 'SUSPICIOUS_ACCURACY'
                    }
                else:
                    return {
                        'accuracy': accuracy,
                        'auc': roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]),
                        'samples': len(X),
                        'target': target_col,
                        'top_feature': feature_cols[np.argmax(model.feature_importances_)],
                        'top_importance': np.max(model.feature_importances_),
                        'issue': None
                    }
        
        return {'accuracy': None, 'issue': 'NO_SUITABLE_TARGET'}
        
    except Exception as e:
        return {'accuracy': None, 'issue': f'ERROR: {str(e)}'}

# Main analysis
projects = {}
for root, dirs, files in os.walk(quality_attributes_path):
    for dir_name in dirs:
        if any(char.isdigit() for char in dir_name):
            project_path = os.path.join(root, dir_name)
            
            # Load all versions for this project
            project_files = []
            for file in os.listdir(project_path):
                if file.endswith('.csv'):
                    csv_file = os.path.join(project_path, file)
                    try:
                        df = pd.read_csv(csv_file)
                        version = os.path.basename(csv_file).replace('.csv', '')
                        df['Version'] = version
                        df['Project'] = dir_name
                        df['File_Source'] = csv_file
                        project_files.append(df)
                    except Exception as e:
                        print(f"   ❌ Error loading {file}: {e}")
            
            if project_files:
                combined_project = pd.concat(project_files, ignore_index=True)
                projects[dir_name] = combined_project
                print(f"✅ {dir_name}: {combined_project.shape}")

# Perform ML analysis for each project
print("\n🤖 PERFORMING ML ANALYSIS FOR EACH PROJECT...")
print("-" * 50)

ml_results = {}
for project_name, project_data in projects.items():
    print(f"🔍 Analyzing: {project_name}...")
    result = safe_ml_analysis(project_data, project_name)
    ml_results[project_name] = result

# Display results and find best valid project
print("\n" + "="*70)
print("🏆 FINAL RESULTS - NO HARDCODED VALUES")
print("="*70)

valid_projects = []
suspicious_projects = []

for project_name, results in ml_results.items():
    if results.get('accuracy') is not None:
        if results.get('issue') == 'SUSPICIOUS_ACCURACY':
            suspicious_projects.append((project_name, results['accuracy']))
            print(f"🚨 {project_name}: SUSPICIOUS Accuracy: {results['accuracy']:.2%}")
        else:
            valid_projects.append((project_name, results['accuracy']))
            print(f"✅ {project_name}:")
            print(f"   Accuracy: {results['accuracy']:.2%}")
            print(f"   AUC: {results.get('auc', 'N/A'):.2%}")
            print(f"   Samples: {results.get('samples', 'N/A'):,}")
            print(f"   Target: {results.get('target', 'N/A')}")
            print(f"   Top Feature: {results.get('top_feature', 'N/A')}")
    else:
        print(f"❌ {project_name}: {results.get('issue', 'Failed')}")

# Find best valid project
if valid_projects:
    best_project, best_accuracy = max(valid_projects, key=lambda x: x[1])
    print(f"\n🎯 BEST VALID PROJECT: {best_project}")
    print(f"   Accuracy: {best_accuracy:.2%}")
    print(f"   Recommendation: Use this project for your final analysis")
    
    # Save the best project's data for final model
    best_data = projects[best_project]
    best_result = ml_results[best_project]
    
    print(f"\n💾 You can now build your final model using: {best_project}")
    
else:
    print("\n❌ No valid projects found for final analysis")
    if suspicious_projects:
        print("💡 Suspicious projects (potential data leakage):")
        for project, acc in suspicious_projects:
            print(f"   {project}: {acc:.2%}")

# Create visualization of all results
plt.figure(figsize=(12, 8))

project_names = []
accuracies = []
colors = []

for project_name, results in ml_results.items():
    if results.get('accuracy') is not None:
        project_names.append(project_name)
        accuracies.append(results['accuracy'])
        if results.get('issue') == 'SUSPICIOUS_ACCURACY':
            colors.append('red')
        else:
            colors.append('green')

if project_names:
    plt.barh(project_names, accuracies, color=colors)
    plt.xlabel('Accuracy')
    plt.title('Project Performance Comparison\n(Red = Suspicious, Green = Valid)')
    plt.axvline(x=0.95, color='orange', linestyle='--', label='Suspicious Threshold (95%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('project_comparison_dynamic.png', dpi=300, bbox_inches='tight')
    plt.show()

print(f"\n✅ DYNAMIC ANALYSIS COMPLETED!")
print(f"📊 Total projects analyzed: {len(projects)}")
print(f"✅ Valid projects: {len(valid_projects)}")
print(f"🚨 Suspicious projects: {len(suspicious_projects)}")