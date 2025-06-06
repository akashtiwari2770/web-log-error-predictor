import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier

# 1. Load data
df = pd.read_csv(
    r"C:\Users\Akash tiwari\PycharmProjects\PythonProject\Web_Logs_Error_Prediction_Using_LGBM\Data\logs\system_logs.csv"
)

# 2. Drop unnecessary columns and missing values
df = df.drop(columns=['source_file', 'sys-thermal'], errors='ignore')
df = df.dropna()

# 3. Create target column (1 = error, 2 = non-error)
df['is_error'] = df['server-up'].apply(lambda x: 1 if x == 1 else 0)

# 4. Define features and target
X = df.drop(columns=['timestamp','server-up', 'is_error'])
y = df['is_error']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 6. Define preprocessing
numeric_features = X.columns.tolist()
preprocessor = ColumnTransformer([('num', StandardScaler(), numeric_features)])


# ========== 2️⃣ Class-Weighted Model ==========
print("\n🔁 Training Class-Weighted Model...")

#  7. Compute class weights manually
classes = sorted(y.unique())
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}

model_weighted = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        class_weight=class_weight_dict
    ))
])

model_weighted.fit(X_train, y_train)
y_pred_weighted = model_weighted.predict(X_test)

# 9. Evaluation
print("\n Class-Weighted Model Report:")
print(classification_report(y_test, y_pred_weighted))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_weighted))

# 10. Save the model
output_path = r"model_class_weight.pkl"
joblib.dump(model_weighted, output_path)
print(f"\n Model saved at: {output_path}")
