import pandas as pd
import numpy as np
import glob
from pathlib import Path

def get_risk_category(score):
    if score == 0:
        return 'Zero (0)', 0
    elif 0 < score <= 100:
        return 'Mild (1-100)', 1
    elif 100 < score <= 300:
        return 'Moderate (101-300)', 2
    elif 300 < score <= 400:
        return 'Mod-High (301-400)', 3
    elif 400 < score <= 1000:
        return 'Severe (401-1000)', 4
    else:
        return 'Extensive (1000+)', 5

csv_files = [
    r"Results\Agaston_results\Agaston Results (Unseen Data)\A1\agatston_comparison_a1.csv",
    r"Results\Agaston_results\Agaston Results (Unseen Data)\A3\agatston_comparison_a3.csv",
    r"Results\Agaston_results\TrainVal_Experiment\agatston_comparison_a1.csv",
    r"Results\Agaston_results\TrainVal_Experiment\agatston_comparison_a3.csv"
]

results = {}

for f in csv_files:
    if not Path(f).exists():
        print(f"Skipping {f}, not found.")
        continue
        
    df = pd.read_csv(f)
    
    # Update categories
    true_cats = df['XML_Agatston'].apply(get_risk_category)
    model_cats = df['Model_Agatston'].apply(get_risk_category)
    
    df['True_Risk_Category'] = [x[0] for x in true_cats]
    df['True_Cat_Idx'] = [x[1] for x in true_cats]
    
    df['Model_Risk_Category'] = [x[0] for x in model_cats]
    df['Model_Cat_Idx'] = [x[1] for x in model_cats]
    
    def get_agreement(row):
        if row['True_Cat_Idx'] == row['Model_Cat_Idx']:
            return 'Match'
        elif row['Model_Cat_Idx'] > row['True_Cat_Idx']:
            return 'Overestimated'
        else:
            return 'Underestimated'
            
    df['Category_Agreement'] = df.apply(get_agreement, axis=1)
    
    # Save back
    df.to_csv(f, index=False)
    
    # Calculate stats
    total = len(df)
    correct = (df['Category_Agreement'] == 'Match').sum()
    acc = correct / total * 100
    
    mean_bias = df.groupby('True_Risk_Category')['Error'].mean()
    
    results[f] = {
        'accuracy': acc,
        'correct': correct,
        'total': total,
        'bias': mean_bias.to_dict()
    }
    
print("--- SUMMARY ---")
for k, v in results.items():
    print(f"\nFile: {k}")
    print(f"Accuracy: {v['accuracy']:.1f}% ({v['correct']}/{v['total']})")
    print("Mean Bias per Category:")
    for cat, bias in v['bias'].items():
        print(f"  {cat}: {bias:.1f}")

# Also update the systematic bias summary if needed
sys_bias_file = r"Results\Agaston_results\Agaston Results (Unseen Data)\investigation\systematic_bias_summary.csv"
if Path(sys_bias_file).exists():
    a1_f = r"Results\Agaston_results\Agaston Results (Unseen Data)\A1\agatston_comparison_a1.csv"
    a3_f = r"Results\Agaston_results\Agaston Results (Unseen Data)\A3\agatston_comparison_a3.csv"
    
    df_a1 = pd.read_csv(a1_f)
    df_a3 = pd.read_csv(a3_f)
    
    bias_a1 = df_a1.groupby('True_Risk_Category')['Error'].mean()
    bias_a3 = df_a3.groupby('True_Risk_Category')['Error'].mean()
    
    bias_df = pd.DataFrame({'A1_Mean_Bias': bias_a1, 'A3_Mean_Bias': bias_a3}).reset_index()
    # Sort logically
    cat_order = ['Zero (0)', 'Mild (1-100)', 'Moderate (101-300)', 'Mod-High (301-400)', 'Severe (401-1000)', 'Extensive (1000+)']
    bias_df['True_Risk_Category'] = pd.Categorical(bias_df['True_Risk_Category'], categories=cat_order, ordered=True)
    bias_df = bias_df.sort_values('True_Risk_Category')
    
    bias_df.to_csv(sys_bias_file, index=False)
    print("\nUpdated systematic_bias_summary.csv")
