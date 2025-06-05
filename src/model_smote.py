import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier

#  1. Load data
df = pd.read_csv(
    r"C:\Users\Akash tiwari\PycharmProjects\PythonProject\Web_Logs_Error_Prediction_Using_LGBM\Data\logs\system_logs.csv"
)

#  2. Drop missing values and columns
df = df.drop(columns=['source_file','sys-thermal'])
df = df.dropna()

#  3. Create target column
df['is_error'] = df['server-up'].apply(lambda x: 1 if x == 1 else 0)


#  4. Define features and target
X = df.drop(columns=['server-up', 'is_error'])
y = df['is_error']

#   5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

#   6. Define preprocessing
numeric_features = X.columns.tolist()
preprocessor = ColumnTransformer([('num', StandardScaler(), numeric_features)])


# ========== 1️ SMOTE-Based Model ==========
print("\n Training SMOTE-based Model...")

smote = SMOTE(random_state=42)

model_smote = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42
    ))
])

model_smote.fit(X_train, y_train)
y_pred_smote = model_smote.predict(X_test)

print("\n SMOTE Model Report:")
print(classification_report(y_test, y_pred_smote))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_smote))

#  7. Save
joblib.dump(model_smote, r'C:\Users\Akash tiwari\PycharmProjects\PythonProject\Web_Logs_Error_Prediction_Using_LGBM\Models\model_smote.pkl')