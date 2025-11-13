import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np

# Your dataset path
dataset_path = "D:\\PROJECTS\\Quality Automation Using ML"

print("🔍 Analyzing your dataset structure...")
print("=" * 50)

# Check if path exists
if os.path.exists(dataset_path):
    print("✅ Path found!")
    
    # List all projects
    projects = os.listdir(dataset_path)
    print(f"📂 Found {len(projects)} projects/folders:")
    
    for i, project in enumerate(projects, 1):
        project_path = os.path.join(dataset_path, project)
        if os.path.isdir(project_path):
            print(f"   {i}. {project}/ (folder)")
        else:
            print(f"   {i}. {project} (file)")
    
    # Analyze the main project in detail
    if projects:
        main_project = projects[0]
        main_project_path = os.path.join(dataset_path, main_project)
        
        print(f"\n🔍 Detailed analysis of: {main_project}/")
        print("-" * 40)
        
        if os.path.isdir(main_project_path):
            # List all files in the project
            project_files = os.listdir(main_project_path)
            csv_files = [f for f in project_files if f.endswith('.csv')]
            
            print(f"Total files: {len(project_files)}")
            print(f"CSV files: {len(csv_files)}")
            
            if csv_files:
                print(f"\n📊 CSV files found:")
                for csv_file in csv_files:
                    file_path = os.path.join(main_project_path, csv_file)
                    try:
                        # Try to read the CSV to get info
                        df = pd.read_csv(file_path)
                        print(f"   ✅ {csv_file}:")
                        print(f"      Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                        
                        # Show first few column names
                        cols_preview = list(df.columns)[:6]
                        if len(df.columns) > 6:
                            cols_preview.append(f"...+{len(df.columns)-6} more")
                        print(f"      Columns: {cols_preview}")
                        
                        # Check for potential target columns
                        target_keywords = ['bug', 'defect', 'error', 'failure', 'problematic', 'issue']
                        target_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in target_keywords)]
                        if target_cols:
                            print(f"      🎯 Potential targets: {target_cols}")
                            
                    except Exception as e:
                        print(f"   ❌ {csv_file}: Error reading")
            else:
                print("   No CSV files found!")
        else:
            print("   This is not a directory!")
    
else:
    print("❌ Path not found!")

print("\n" + "="*60)
print("📊 LOADING AND PREVIEWING MAIN DATASETS")
print("="*60)

# Load the main datasets
main_project_path = os.path.join(dataset_path, "Software code quality and source code metrics dataset")

# Load versions.csv (this seems to be your main data)
versions_df = pd.read_csv(os.path.join(main_project_path, "versions.csv"))
repositories_df = pd.read_csv(os.path.join(main_project_path, "repositories.csv"))
attribute_details_df = pd.read_csv(os.path.join(main_project_path, "attribute-details.csv"))

print("\n1. VERSIONS DATASET (Main Data):")
print("   Shape:", versions_df.shape)
print("\n   First 5 rows:")
print(versions_df.head())
print("\n   Column names:")
for col in versions_df.columns:
    print(f"   - {col}")

print("\n2. REPOSITORIES DATASET:")
print("   Shape:", repositories_df.shape)
print("\n   First 3 rows:")
print(repositories_df.head(3))

print("\n3. ATTRIBUTE DETAILS DATASET:")
print("   Shape:", attribute_details_df.shape)
print("\n   Content preview:")
print(attribute_details_df.head())