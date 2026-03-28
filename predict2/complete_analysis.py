import os
import plistlib
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================
# STEP 1 — CORRECT AGATSTON SCORE PARSING
# ============================================================
print("="*60)
print("STEP 1: Parsing Agatston Scores")
print("="*60)

XML_PATH = r"D:\COCA_data\calcium_xml\calcium_xml"

def parse_agatston(xml_file):
    """
    Real Agatston score calculation:
    Score = Area(mm2) x Density_factor
    Density factors: HU 130-199=1, 200-299=2, 300-399=3, 400+=4
    """
    try:
        with open(xml_file, 'rb') as f:
            plist = plistlib.load(f)

        total_agatston = 0
        total_volume = 0
        lesion_count = 0

        for img in plist.get("Images", []):
            for roi in img.get("ROIs", []):
                area = roi.get("Area", 0)
                max_hu = roi.get("Max", 0)
                mean_hu = roi.get("Mean", max_hu)

                if area <= 0:
                    continue

                # Density factor based on peak HU
                if max_hu >= 400:
                    density_factor = 4
                elif max_hu >= 300:
                    density_factor = 3
                elif max_hu >= 200:
                    density_factor = 2
                elif max_hu >= 130:
                    density_factor = 1
                else:
                    continue

                total_agatston += area * density_factor
                total_volume += area
                lesion_count += 1

        return total_agatston, total_volume, lesion_count

    except Exception as e:
        print(f"  Error: {e}")
        return 0, 0, 0

data = []
for file in sorted(os.listdir(XML_PATH)):
    if not file.endswith(".xml") or file == "000.DS_Store":
        continue
    patient_id = file.replace(".xml", "")
    xml_file = os.path.join(XML_PATH, file)
    agatston, volume, lesions = parse_agatston(xml_file)
    data.append({
        "patient_id": patient_id,
        "agatston_score": round(agatston, 2),
        "calcium_volume": round(volume, 2),
        "lesion_count": lesions
    })

agatston_df = pd.DataFrame(data)

# Agatston categories
def categorize(score):
    if score == 0: return "None"
    elif score <= 10: return "Minimal"
    elif score <= 100: return "Mild"
    elif score <= 400: return "Moderate"
    else: return "Severe"

agatston_df['agatston_category'] = agatston_df['agatston_score'].apply(categorize)
agatston_df.to_csv(r"D:\COCA_data\agatston_correct.csv", index=False)

print(f"Parsed {len(agatston_df)} patients")
print(f"Score range: {agatston_df['agatston_score'].min():.1f} - {agatston_df['agatston_score'].max():.1f}")
print(f"\nCategory distribution:")
print(agatston_df['agatston_category'].value_counts())
print(agatston_df[['patient_id','agatston_score','agatston_category']].head(10))

# ============================================================
# STEP 2 — MERGE WITH FEATURES
# ============================================================
print("\n" + "="*60)
print("STEP 2: Merging Features with Agatston Scores")
print("="*60)

features_df = pd.read_csv(r"D:\COCA_data\features.csv")

# Get patient_id from scan_index
scan_index = pd.read_csv(r"D:\COCA_data\processed\tables\scan_index.csv")
print("Scan index columns:", scan_index.columns.tolist())

# Merge features with scan_index to get patient_id
merged = features_df.merge(scan_index[['scan_id','patient_id']], on='scan_id', how='left')

# Merge with agatston scores
merged['patient_id'] = merged['patient_id'].astype(str)
agatston_df['patient_id'] = agatston_df['patient_id'].astype(str)
final_df = merged.merge(agatston_df, on='patient_id', how='left')

print(f"Final dataset: {final_df.shape}")
print(f"Patients with Agatston scores: {final_df['agatston_score'].notna().sum()}")
print(final_df[['patient_id','agatston_score','agatston_category']].head(10))

final_df.to_csv(r"D:\COCA_data\final_merged.csv", index=False)

# ============================================================
# STEP 3 — FEATURE EXTRACTION STATS
# ============================================================
print("\n" + "="*60)
print("STEP 3: Dataset Statistics")
print("="*60)

feature_cols = [c for c in features_df.columns if c != 'scan_id']
print(f"Total features extracted: {len(feature_cols)}")
print(f"Total patients: {len(final_df)}")
print(f"\nFeature categories:")
for prefix in ['shape', 'firstorder', 'glcm', 'glszm', 'glrlm', 'ngtdm']:
    count = len([c for c in feature_cols if prefix in c])
    print(f"  {prefix}: {count} features")

# ============================================================
# STEP 4 — SPEARMAN CORRELATION
# ============================================================
print("\n" + "="*60)
print("STEP 4: Spearman Correlation with Agatston Score")
print("="*60)

valid_df = final_df.dropna(subset=['agatston_score'])
feature_cols_clean = [c for c in feature_cols if c in valid_df.columns]

correlations = []
for feat in feature_cols_clean:
    try:
        vals = valid_df[feat].fillna(0)
        scores = valid_df['agatston_score']
        corr, pval = stats.spearmanr(vals, scores)
        correlations.append({
            'feature': feat,
            'spearman_r': round(corr, 4),
            'p_value': round(pval, 4),
            'abs_corr': abs(corr),
            'significant': pval < 0.05
        })
    except:
        pass

corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)
corr_df.to_csv(r"D:\COCA_data\spearman_results.csv", index=False)

print(f"Significant features (p<0.05): {corr_df['significant'].sum()}")
print("\nTop 15 features by Spearman correlation:")
print(corr_df.head(15)[['feature','spearman_r','p_value','significant']].to_string())

# ============================================================
# STEP 5 — KRUSKAL-WALLIS TEST
# ============================================================
print("\n" + "="*60)
print("STEP 5: Kruskal-Wallis Test Across Agatston Categories")
print("="*60)

kruskal_results = []
categories = valid_df['agatston_category'].unique()
print(f"Categories present: {categories}")

for feat in feature_cols_clean:
    try:
        groups = [valid_df[valid_df['agatston_category']==cat][feat].dropna().values
                 for cat in categories]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            stat, pval = stats.kruskal(*groups)
            kruskal_results.append({
                'feature': feat,
                'kruskal_stat': round(stat, 4),
                'p_value': round(pval, 4),
                'significant': pval < 0.05
            })
    except:
        pass

kruskal_df = pd.DataFrame(kruskal_results).sort_values('p_value')
kruskal_df.to_csv(r"D:\COCA_data\kruskal_results.csv", index=False)

print(f"Significant features (p<0.05): {kruskal_df['significant'].sum()}")
print("\nTop 10 significant features:")
print(kruskal_df.head(10)[['feature','kruskal_stat','p_value']].to_string())

# ============================================================
# STEP 6 — CLUSTERING
# ============================================================
print("\n" + "="*60)
print("STEP 6: Clustering")
print("="*60)

X = valid_df[feature_cols_clean].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for dimensionality reduction
pca = PCA(n_components=min(10, len(feature_cols_clean)))
X_pca = pca.fit_transform(X_scaled)
print(f"PCA variance explained (10 components): {pca.explained_variance_ratio_.sum():.3f}")

# Find best K
print("\nFinding best number of clusters:")
best_k, best_score = 2, -1
for k in range(2, min(6, len(valid_df))):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    if len(set(labels)) > 1:
        score = silhouette_score(X_pca, labels)
        print(f"  K={k}: Silhouette={score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k

print(f"\nBest K={best_k} (Silhouette={best_score:.3f})")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
valid_df = valid_df.copy()
valid_df['cluster'] = kmeans.fit_predict(X_pca)

print("\nCluster distribution:")
print(valid_df['cluster'].value_counts())
print("\nCluster vs Agatston category:")
print(pd.crosstab(valid_df['cluster'], valid_df['agatston_category']))

# ============================================================
# STEP 7 — VISUALIZATIONS
# ============================================================
print("\n" + "="*60)
print("STEP 7: Creating Visualizations")
print("="*60)

# Color maps
cat_colors = {'None':'green','Minimal':'blue','Mild':'orange',
              'Moderate':'red','Severe':'darkred','Unknown':'gray'}

# --- Plot 1: PCA colored by Agatston category ---
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cats = valid_df['agatston_category'].fillna('Unknown')
for cat, color in cat_colors.items():
    mask = cats == cat
    if mask.any():
        axes[0].scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                       c=color, label=cat, alpha=0.8, s=120, edgecolors='black', linewidth=0.5)
axes[0].set_title('PCA — Colored by Agatston Category', fontsize=13, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% variance)')
axes[0].set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% variance)')
axes[0].legend(title='Agatston Category')
axes[0].grid(True, alpha=0.3)

# Colored by cluster
scatter = axes[1].scatter(X_pca2[:, 0], X_pca2[:, 1],
                          c=valid_df['cluster'], cmap='Set1',
                          alpha=0.8, s=120, edgecolors='black', linewidth=0.5)
axes[1].set_title(f'PCA — K-Means Clusters (K={best_k})', fontsize=13, fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% variance)')
axes[1].set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.colorbar(scatter, ax=axes[1], label='Cluster')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"D:\COCA_data\pca_clustering.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pca_clustering.png")

# --- Plot 2: Top 15 Spearman correlations ---
fig, ax = plt.subplots(figsize=(12, 8))
top15 = corr_df.head(15)
colors_bar = ['green' if r > 0 else 'red' for r in top15['spearman_r']]
bars = ax.barh(range(len(top15)), top15['spearman_r'], color=colors_bar, alpha=0.7)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels([f.replace('original_','') for f in top15['feature']], fontsize=9)
ax.set_xlabel('Spearman Correlation Coefficient')
ax.set_title('Top 15 Features Correlated with Agatston Score', fontsize=13, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(r"D:\COCA_data\spearman_plot.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: spearman_plot.png")

# --- Plot 3: Agatston score distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(valid_df['agatston_score'], bins=10, color='steelblue',
             edgecolor='black', alpha=0.7)
axes[0].set_title('Agatston Score Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Agatston Score')
axes[0].set_ylabel('Count')
axes[0].grid(True, alpha=0.3)

cat_counts = valid_df['agatston_category'].value_counts()
axes[1].bar(cat_counts.index, cat_counts.values,
            color=[cat_colors.get(c,'gray') for c in cat_counts.index],
            edgecolor='black', alpha=0.8)
axes[1].set_title('Patients per Agatston Category', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Category')
axes[1].set_ylabel('Count')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(r"D:\COCA_data\agatston_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: agatston_distribution.png")

# --- Plot 4: Feature correlation heatmap (top 12) ---
top12_feats = corr_df.head(12)['feature'].tolist()
if len(top12_feats) >= 3:
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = valid_df[top12_feats].corr()
    feat_labels = [f.replace('original_','') for f in top12_feats]
    corr_matrix.columns = feat_labels
    corr_matrix.index = feat_labels
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Heatmap (Top 12 Features)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(r"D:\COCA_data\correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: correlation_heatmap.png")

# ============================================================
# STEP 8 — RANDOM FOREST + FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*60)
print("STEP 8: Random Forest Feature Importance")
print("="*60)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, valid_df['agatston_score'])

importances = pd.Series(rf.feature_importances_, index=feature_cols_clean)
top20 = importances.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(12, 8))
top20.plot(kind='barh', ax=ax, color='steelblue', alpha=0.8, edgecolor='black')
ax.set_title('Top 20 Features by Random Forest Importance', fontsize=13, fontweight='bold')
ax.set_xlabel('Feature Importance')
ax.set_yticklabels([f.replace('original_','') for f in top20.index], fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(r"D:\COCA_data\feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")

top20.to_csv(r"D:\COCA_data\top_features.csv", header=['importance'])
print("\nTop 10 most important features:")
print(top20.head(10).to_string())

# ============================================================
# STEP 9 — FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("STEP 9: FINAL SUMMARY")
print("="*60)
print(f"✅ Patients processed: {len(valid_df)}")
print(f"✅ Features extracted: {len(feature_cols_clean)}")
print(f"✅ Agatston score range: {valid_df['agatston_score'].min():.1f} - {valid_df['agatston_score'].max():.1f}")
print(f"✅ Significant Spearman features: {corr_df['significant'].sum()}")
print(f"✅ Significant Kruskal-Wallis features: {kruskal_df['significant'].sum()}")
print(f"✅ Best clustering K: {best_k} (Silhouette: {best_score:.3f})")
print(f"\nFiles saved:")
print(f"  D:/COCA_data/agatston_correct.csv")
print(f"  D:/COCA_data/spearman_results.csv")
print(f"  D:/COCA_data/kruskal_results.csv")
print(f"  D:/COCA_data/top_features.csv")
print(f"  D:/COCA_data/pca_clustering.png")
print(f"  D:/COCA_data/spearman_plot.png")
print(f"  D:/COCA_data/agatston_distribution.png")
print(f"  D:/COCA_data/correlation_heatmap.png")
print(f"  D:/COCA_data/feature_importance.png")
print("\n🎉 PREDICT2 Analysis Complete!")