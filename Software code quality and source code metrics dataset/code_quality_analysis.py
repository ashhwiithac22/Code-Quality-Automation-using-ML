import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

print("🚀 DISCOVERED DETAILED CODE METRICS DATASET!")
print("=" * 60)

dataset_path = "D:\\PROJECTS\\Quality Automation Using ML"
quality_attributes_path = os.path.join(dataset_path, "Software code quality and source code metrics dataset", "quality_attributes")

# Explore the structure
print("📁 QUALITY ATTRIBUTES FOLDER STRUCTURE:")
print("-" * 40)

all_csv_files = []
for root, dirs, files in os.walk(quality_attributes_path):
    for file in files:
        if file.endswith('.csv'):
            full_path = os.path.join(root, file)
            all_csv_files.append(full_path)
            # Show sample structure
            if len(all_csv_files) <= 3:  # Show first 3 files
                df_sample = pd.read_csv(full_path)
                print(f"📊 {file}: {df_sample.shape} - Columns: {list(df_sample.columns)[:5]}...")

print(f"\n✅ Found {len(all_csv_files)} detailed code metric files!")

# Load and combine ALL the detailed metrics
print("\n🔍 COMBINING ALL DETAILED CODE METRICS...")
print("-" * 40)

all_detailed_data = []

for csv_file in all_csv_files:
    try:
        df = pd.read_csv(csv_file)
        
        # Extract project name and version from file path
        file_parts = csv_file.split('\\')
        project_name = file_parts[-3]  # Project folder name
        version_name = file_parts[-1].replace('.csv', '')  # File name without .csv
        
        # Add identifier columns
        df['Project'] = project_name
        df['Version'] = version_name
        
        all_detailed_data.append(df)
        
    except Exception as e:
        print(f"❌ Error reading {csv_file}: {e}")

# Combine all data
if all_detailed_data:
    combined_detailed_df = pd.concat(all_detailed_data, ignore_index=True)
    print(f"✅ COMBINED DETAILED DATASET: {combined_detailed_df.shape}")
    print(f"📊 Columns: {list(combined_detailed_df.columns)}")
    
    # Check for target variables
    target_keywords = ['bug', 'defect', 'problematic', 'error', 'quality', 'smell']
    potential_targets = [col for col in combined_detailed_df.columns 
                        if any(keyword in str(col).lower() for keyword in target_keywords)]
    
    print(f"🎯 Potential targets: {potential_targets}")
    
    # Show sample of the data
    print(f"\n📋 SAMPLE DATA:")
    print(combined_detailed_df.head(3))
else:
    print("❌ No detailed data files could be loaded!")
    combined_detailed_df = None

# Compare datasets
print("\n" + "="*60)
print("📊 DATASET COMPARISON")
print("=" * 60)

print("1. OLD DATASET (versions.csv):")
print("   - Summary statistics per version")
print("   - 61 rows, 12 columns")
print("   - Limited features")

if combined_detailed_df is not None:
    print(f"\n2. NEW DETAILED DATASET (quality_attributes):")
    print(f"   - Detailed code metrics per file/class")
    print(f"   - {combined_detailed_df.shape[0]} rows, {combined_detailed_df.shape[1]} columns")
    print(f"   - Rich features for better predictions")
    
    # Quick ML test on detailed data
    print(f"\n🤖 QUICK ML TEST ON DETAILED DATA:")
    
    # Find numeric columns for features
    numeric_cols = combined_detailed_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove identifier columns
    feature_cols = [col for col in numeric_cols if col not in ['Project', 'Version']]
    
    if len(feature_cols) >= 5:  # If we have enough features
        # Find a target (use first numeric column that's not an ID)
        potential_targets = [col for col in feature_cols if 'id' not in col.lower() and 'no' not in col.lower()]
        
        if potential_targets:
            target_col = potential_targets[0]
            X_detailed = combined_detailed_df[feature_cols].fillna(0)
            y_detailed = combined_detailed_df[target_col]
            
            # Convert to binary classification if needed
            if y_detailed.nunique() > 10:
                y_detailed = (y_detailed > y_detailed.median()).astype(int)
            
            # Test ML performance
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            n_folds = min(5, len(X_detailed) // 10)
            n_folds = max(2, n_folds)
            
            try:
                cv_scores = cross_val_score(rf, X_detailed, y_detailed, cv=n_folds)
                print(f"   ✅ Detailed data CV Accuracy: {cv_scores.mean():.4f} ({n_folds}-fold)")
                print(f"   🎯 Using target: {target_col}")
            except Exception as e:
                print(f"   ❌ ML test failed: {e}")
    
    print(f"\n🏆 RECOMMENDATION: USE THE DETAILED DATASET!")
    print("   Reason: Much richer features for better code quality predictions")