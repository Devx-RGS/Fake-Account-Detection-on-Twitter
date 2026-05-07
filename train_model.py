import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

print("="*55)
print("TWITTER BOT DETECTION - FULL DATASET (11,826 users)")
print("="*55)

# ── Load full dataset ─────────────────────────────────────
df = pd.read_csv(r'D:\Fake Account Detection in Twitter\twibot_full_features.csv')
df = df.drop(columns=['verified'], errors='ignore')   # remove biased feature

print(f"\nDataset shape : {df.shape}")
print(f"Bots          : {(df['label']==1).sum()}  ({(df['label']==1).mean()*100:.1f}%)")
print(f"Humans        : {(df['label']==0).sum()}  ({(df['label']==0).mean()*100:.1f}%)")

# ── Features ──────────────────────────────────────────────
X = df.drop(['id', 'label'], axis=1)
y = df['label']
feature_names = X.columns.tolist()

print(f"\nFeatures used : {len(feature_names)}")
for i, f in enumerate(feature_names, 1):
    print(f"  {i:2}. {f}")

# ── Split & Scale ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── Train 6 Models ────────────────────────────────────────
print("\n" + "="*55)
print("MODEL COMPARISON")
print("="*55)
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-"*65)

models = {
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting'  : GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree'      : DecisionTreeClassifier(max_depth=10, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes'        : GaussianNB(),
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    results[name] = {
        'model': model, 'y_pred': y_pred,
        'accuracy' : accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall'   : recall_score(y_test, y_pred),
        'f1'       : f1_score(y_test, y_pred),
    }
    r = results[name]
    print(f"{name:<25} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")

best_name = max(results, key=lambda x: results[x]['f1'])
print(f"\nBEST MODEL: {best_name}  (F1={results[best_name]['f1']:.4f})")

# ── Confusion Matrix ──────────────────────────────────────
print("\n" + "="*55)
print(f"CONFUSION MATRIX  [{best_name}]")
print("="*55)
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
TN, FP, FN, TP = cm.ravel()

print(f"\n                  PREDICTED")
print(f"                  Human    Bot")
print(f"  ACTUAL  Human  [ {TN:>4}   {FP:>4} ]")
print(f"          Bot    [ {FN:>4}   {TP:>4} ]")
print(f"\n  True  Negatives (Human->Human) : {TN}")
print(f"  False Positives (Human->Bot)   : {FP}  << humans mislabelled")
print(f"  False Negatives (Bot->Human)   : {FN}  << bots that slipped through!")
print(f"  True  Positives (Bot->Bot)     : {TP}")
print(f"\n  Bot   Recall    : {TP/(TP+FN)*100:.1f}%  (how many bots we caught)")
print(f"  Human Precision : {TN/(TN+FP)*100:.1f}%  (when we say human, how often right)")

# ── Classification Report ─────────────────────────────────
print(f"\n  Classification Report ({best_name}):")
print(classification_report(y_test, results[best_name]['y_pred'],
                            target_names=['Human', 'Bot']))

# ── Feature Importance ────────────────────────────────────
print("="*55)
print("TOP 10 FEATURE IMPORTANCES (Random Forest)")
print("="*55)
rf = results['Random Forest']['model']
imp = pd.DataFrame({'Feature': feature_names,
                    'Importance': rf.feature_importances_})
imp = imp.sort_values('Importance', ascending=False).reset_index(drop=True)
for i, row in imp.head(10).iterrows():
    bar = "#" * int(row['Importance'] * 100)
    print(f"  {i+1:2}. {row['Feature']:<30} {row['Importance']:.4f}  {bar}")

# ── Save best model ───────────────────────────────────────
best_model = results[best_name]['model']
with open(r'D:\Fake Account Detection in Twitter\bot_detector_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open(r'D:\Fake Account Detection in Twitter\bot_detector_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"\nModel saved: bot_detector_model.pkl")
print(f"Scaler saved: bot_detector_scaler.pkl")
print("="*55)
