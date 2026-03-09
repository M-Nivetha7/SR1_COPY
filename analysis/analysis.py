import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import datetime

print("=" * 60)
print("🚀 SDSS SkyServer Astronomical Classification")
print("=" * 60)

# Load the dataset
print("\n📂 Loading dataset...")
df = pd.read_csv("Skyserver_SQL2_27_2018 6_51_39 PM.csv")
print(f"✅ Dataset loaded: {df.shape[0]} observations, {df.shape[1]} features")

# Display class distribution
class_counts = df['class'].value_counts()
print("\n📊 Class Distribution:")
for cls, count in class_counts.items():
    print(f"   {cls}: {count} ({count/len(df)*100:.1f}%)")

# Drop unnecessary columns
df = df.drop(['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid'], axis=1)

# Encode target labels
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])
class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"\n🔤 Class mapping: {class_mapping}")

# Prepare features and target
X = df.drop('class', axis=1)
y = df['class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📈 Train-Test Split:")
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Testing set: {X_test.shape[0]} samples")

# Train Random Forest model
print("\n🌲 Training Random Forest Classifier (n_estimators=200)...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracies
train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"   Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Testing Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")

# Get classification report
report = classification_report(y_test, y_pred, output_dict=True, target_names=le.classes_)

# Get confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Feature importance
importance = model.feature_importances_
feature_importance = pd.Series(importance, index=X.columns).sort_values(ascending=False)

print("\n🔍 Top 5 Most Important Features:")
for i, (feature, imp) in enumerate(feature_importance.head(5).items()):
    print(f"   {i+1}. {feature}: {imp:.4f}")

# Create results directory
os.makedirs('../results', exist_ok=True)

# Prepare results for JSON
results = {
    'project_info': {
        'name': 'SDSS SkyServer Astronomical Classification',
        'description': 'Classification of celestial objects into Stars, Galaxies, and Quasars using Random Forest',
        'dataset_size': len(df),
        'features': list(X.columns),
        'classes': list(le.classes_),
        'class_counts': class_counts.to_dict(),
        'class_mapping': {str(k): int(v) for k, v in class_mapping.items()}
    },
    'model_info': {
        'algorithm': 'Random Forest Classifier',
        'n_estimators': 200,
        'random_state': 42
    },
    'performance': {
        'training_accuracy': float(train_acc),
        'testing_accuracy': float(test_acc),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    },
    'feature_importance': feature_importance.to_dict(),
    'sample_predictions': [
        {
            'actual': le.inverse_transform([y_test.iloc[i]])[0],
            'predicted': le.inverse_transform([y_pred[i]])[0],
            'confidence': float(max(model.predict_proba(X_test.iloc[i:i+1])[0])),
            'features': X_test.iloc[i].to_dict()
        }
        for i in range(min(10, len(X_test)))
    ],
    'generated_date': datetime.datetime.now().isoformat(),
    'note': 'High accuracy (~99%) is expected because astronomical features (redshift and spectral bands) clearly differentiate stars, galaxies, and quasars in the SDSS dataset.'
}

# Save to JSON
output_path = '../results/results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {output_path}")
print("=" * 60)
