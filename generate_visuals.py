import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

cat_order = ['Zero (0)', 'Mild (1-100)', 'Moderate (101-300)', 'Mod-High (301-400)', 'Severe (401-1000)', 'Extensive (1000+)']

def plot_confusion_matrix(csv_path, out_path, title):
    df = pd.read_csv(csv_path)
    
    true_cat = pd.Categorical(df['True_Risk_Category'], categories=cat_order)
    model_cat = pd.Categorical(df['Model_Risk_Category'], categories=cat_order)
    
    cm = confusion_matrix(true_cat, model_cat, labels=cat_order)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cat_order, yticklabels=cat_order)
    plt.title(title)
    plt.ylabel('True Risk Category')
    plt.xlabel('Predicted Risk Category')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# Unseen Data
unseen_a1_csv = r"Results\Agaston_results\Agaston Results (Unseen Data)\A1\agatston_comparison_a1.csv"
unseen_a3_csv = r"Results\Agaston_results\Agaston Results (Unseen Data)\A3\agatston_comparison_a3.csv"

plot_confusion_matrix(unseen_a1_csv, r"Results\Agaston_results\Agaston Results (Unseen Data)\A1\agatston_confusion_a1.png", "A1 Confusion Matrix (Unseen Data)")
plot_confusion_matrix(unseen_a3_csv, r"Results\Agaston_results\Agaston Results (Unseen Data)\A3\agatston_confusion_a3.png", "A3 Confusion Matrix (Unseen Data)")

# TrainVal
trainval_a1_csv = r"Results\Agaston_results\TrainVal_Experiment\agatston_comparison_a1.csv"
trainval_a3_csv = r"Results\Agaston_results\TrainVal_Experiment\agatston_comparison_a3.csv"

plot_confusion_matrix(trainval_a1_csv, r"Results\Agaston_results\TrainVal_Experiment\agatston_confusion_a1.png", "A1 Confusion Matrix (TrainVal)")
plot_confusion_matrix(trainval_a3_csv, r"Results\Agaston_results\TrainVal_Experiment\agatston_confusion_a3.png", "A3 Confusion Matrix (TrainVal)")

# Accuracy Bar Charts
def plot_accuracy_bar(a1_acc, a3_acc, out_path, title):
    plt.figure(figsize=(6, 5))
    bars = plt.bar(['A1 (Binary)', 'A3 (Soft Coverage)'], [a1_acc, a3_acc], color=['skyblue', 'lightgreen'])
    plt.ylim(0, 100)
    plt.ylabel('Clinical Risk Accuracy (%)')
    plt.title(title)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

plot_accuracy_bar(77.3, 83.3, r"Results\Agaston_results\Agaston Results (Unseen Data)\agatston_comparison_metrics_bar.png", "Risk Category Accuracy (Unseen Data)")
plot_accuracy_bar(70.7, 76.7, r"Results\Agaston_results\TrainVal_Experiment\agatston_comparison_metrics_bar.png", "Risk Category Accuracy (TrainVal)")

print("Visuals updated successfully.")
