import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="System Log Error Predictor", layout="wide")
st.title(" System Log Error Predictor")

# Load both models
model_paths = {

    "SMOTE-based Model": r'C:\Users\Akash tiwari\log prediction model\model_smote.pkl',
                         
    "Class-Weighted Model": r'C:\Users\Akash tiwari\log prediction model\model_class_weight.pkl'
}

model_choice = st.selectbox(" Choose Prediction Model", list(model_paths.keys()))
model = joblib.load(model_paths[model_choice])

# Upload CSV
uploaded_file = st.file_uploader(" Upload system log CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(" Preview of uploaded data:", df.head())

    # Optional target column
    has_target = False
    if 'server-up' in df.columns:
        df['is_error'] = df['server-up'].apply(lambda x: 1 if x == 1 else 0)
        has_target = True
        st.success(" Found target column `server-up`. Added `is_error`.")
    else:
        st.info(" `server-up` column not found. Proceeding without target labels.")

    # Extract preprocessor columns
    preprocessor = model.named_steps['preprocessor']
    numeric_cols, categorical_cols = [], []
    for name, _, cols in preprocessor.transformers_:
        if name == 'num':
            numeric_cols = list(cols)
        elif name == 'cat':
            categorical_cols = list(cols)
    expected_columns = numeric_cols + categorical_cols

    # Prepare input
    X_input = df.copy()
    X_input.drop(columns=[col for col in X_input.columns if col not in expected_columns], inplace=True)
    for col in expected_columns:
        if col not in X_input.columns:
            X_input[col] = 0 if col in numeric_cols else "missing"
    X_input = X_input[expected_columns]

    # Predict
    preds = model.predict(X_input)
    probs = model.predict_proba(X_input)[:, 1]
    df['predicted_error'] = preds
    df['error_probability'] = probs

    # Display results
    st.subheader(" Prediction Results")
    if has_target:
        st.write(df[['is_error', 'predicted_error', 'error_probability']])
    else:
        st.write(df[['predicted_error', 'error_probability']])

    #  Show predicted errors
    with st.expander(" Show Predicted Error Rows Only"):
        error_rows = df[df['predicted_error'] == 1]
        display_cols = ['predicted_error', 'error_probability'] + [
            col for col in expected_columns if col in error_rows.columns
        ]
        st.write(error_rows[display_cols])

    #  Bar chart of predictions
    st.subheader(" Error Prediction Distribution")
    pred_counts = df['predicted_error'].value_counts().rename({0: "Normal", 1: "Error"})
    fig1, ax1 = plt.subplots()
    sns.barplot(x=pred_counts.index, y=pred_counts.values, palette="Set2", ax=ax1)
    ax1.set_ylabel("Count")
    ax1.set_title("Predicted Normal vs Error")
    st.pyplot(fig1)

    #  Feature Importance (LGBM only)
    try:
        lgbm_model = model.named_steps['classifier']
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        importances = lgbm_model.feature_importances_

        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(20)

        st.subheader(" Top 20 Feature Importances")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette="viridis", ax=ax2)
        ax2.set_title("Top Features (LGBM)")
        st.pyplot(fig2)
    except Exception as e:
        st.warning(" Could not display feature importance: " + str(e))

    # Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(" Download Results CSV", csv, file_name="predictions.csv", mime="text/csv")
