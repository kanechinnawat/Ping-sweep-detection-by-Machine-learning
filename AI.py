import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os  
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

save_dir = 'results'
os.makedirs(save_dir, exist_ok=True)
print(f" Images will be saved to folder: {save_dir}/")


# 1. Load data
df = pd.read_csv('ping_sweep_enterprise.csv', skipinitialspace=True) 
df.columns = df.columns.str.strip() 

# 2. Preprocessing
df_clean = df.drop(columns=['timestamp', 'src_ip', 'dst_ip'])

categorical_cols = ['protocol', 'flag']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

X = df_clean.drop(columns=['label'])
y = df_clean['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 3. Model Training & Comparison 
print("\n--- Training Models ---")

# --- Model 1: Random Forest ---
print("1. Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"   -> Random Forest Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")

# --- Model 2: XGBoost ---
print("2. Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"   -> XGBoost Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")

# Selection Logic 
if xgb_acc > rf_acc:
    best_model = xgb_model
    y_pred = xgb_pred
    model_name = "XGBoost"
    print(f"\n WINNER: {model_name} (Accuracy is higher)")
else:
    best_model = rf_model
    y_pred = rf_pred
    model_name = "Random Forest"
    print(f"\n WINNER: {model_name} (Accuracy is higher or equal)")


# 4. Cross Validation
print(f"\nRunning Cross-Validation on {model_name}...")
cv_scores = cross_val_score(best_model, X, y, cv=5)
print(f"Mean CV Accuracy ({model_name}): {cv_scores.mean():.4f}")


# 5. Visualization & Saving
print("\nGenerating and saving plots...")

# --- ส่วนที่ 1: Confusion Matrix Heatmap ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal (0)', 'Attack (1)'],
            yticklabels=['Normal (0)', 'Attack (1)'])

plt.title(f'Confusion Matrix: {model_name}', fontsize=14, weight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()

# SAVE
plt.savefig(f'{save_dir}/1_confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"Saved: {save_dir}/1_confusion_matrix.png")
plt.show()

# --- ส่วนที่ 2: Classification Report Table ---
report_dict = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'], output_dict=True)
metrics_df = pd.DataFrame(report_dict).transpose()

fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')
ax.axis('tight')

table = ax.table(cellText=metrics_df.round(4).values,
                 colLabels=metrics_df.columns,
                 rowLabels=metrics_df.index,
                 cellLoc='center',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#40466e')
        cell.set_text_props(color='white')
    elif col == -1:
        cell.set_text_props(weight='bold')

plt.title(f'Classification Metrics ({model_name})', fontsize=16, weight='bold', y=1.05)

# SAVE
plt.savefig(f'{save_dir}/2_metrics_table.png', dpi=300, bbox_inches='tight')
print(f"Saved: {save_dir}/2_metrics_table.png")
plt.show()

# --- ส่วนที่ 3: Feature Importance Pie Chart ---
importances = best_model.feature_importances_
feature_names = X.columns

df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
df_imp = df_imp.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(9, 9))
explode = [0.05 if i == 0 else 0 for i in range(len(df_imp))]
colors = sns.color_palette('pastel')[0:len(df_imp)]

plt.pie(df_imp['Importance'], 
        labels=df_imp['Feature'], 
        autopct='%1.1f%%',      
        startangle=90,          
        explode=explode,        
        shadow=True,            
        colors=colors,          
        textprops={'fontsize': 12})

plt.title(f'Feature Importance Distribution ({model_name})\n', fontsize=15, weight='bold')

# SAVE
plt.savefig(f'{save_dir}/3_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"Saved: {save_dir}/3_feature_importance.png")
plt.show()

print(f"\n--- Feature Importance Ranking ({model_name}) ---")
for index, row in df_imp.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")

# 6. Deep Dive Visualization
print("\nCreating Deep Dive Visualizations...")

sns.set_style("whitegrid")
df_viz = df.copy() 
df_viz['label_str'] = df_viz['label'].map({0: 'Normal', 1: 'Attack'})

# 6.1: Protocol & Flag Analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.countplot(data=df_viz, x='protocol', hue='label_str', palette='coolwarm', ax=axes[0])
axes[0].set_title('Protocol Frequency: Attack vs Normal', fontsize=14, weight='bold')

top_flags = df_viz['flag'].value_counts().index
sns.countplot(data=df_viz, y='flag', hue='label_str', palette='coolwarm', ax=axes[1], order=top_flags)
axes[1].set_title('TCP/ICMP Flag Distribution', fontsize=14, weight='bold')
plt.tight_layout()

# SAVE
plt.savefig(f'{save_dir}/4_deep_dive_protocol_flag.png', dpi=300, bbox_inches='tight')
print(f"Saved: {save_dir}/4_deep_dive_protocol_flag.png")
plt.show()

# 6.2: Packet Size & TTL Analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(data=df_viz, x='label_str', y='packet_size', hue='label_str', legend=False, palette='Set2', ax=axes[0])
axes[0].set_title('Packet Size Distribution', fontsize=14, weight='bold')

sns.violinplot(data=df_viz, x='label_str', y='ttl', hue='label_str', legend=False, palette='muted', split=True, ax=axes[1])
axes[1].set_title('TTL (Time-To-Live) Distribution', fontsize=14, weight='bold')
plt.tight_layout()

# SAVE
plt.savefig(f'{save_dir}/5_deep_dive_packet_ttl.png', dpi=300, bbox_inches='tight')
print(f"Saved: {save_dir}/5_deep_dive_packet_ttl.png")
plt.show()

print("\n DONE: All processes completed and images saved.")