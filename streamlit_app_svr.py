import json
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


st.set_page_config(page_title="SVR Streamlit App", layout="wide")

st.title("Support Vector Regression (SVR) - Streamlit App")


@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def train_svr(X: np.ndarray, y: np.ndarray, *, kernel: str, C: float, gamma: str, epsilon: float):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2


# Sidebar controls
with st.sidebar:
    st.header("Dataset")

    uploaded = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        accept_multiple_files=False,
        help="CSV must include a target column named 'x'.",
    )

    csv_path = st.text_input("Or use local CSV path", value="test.csv")

    st.divider()
    st.header("Target & Features")

    target_col = st.text_input("Target column", value="x")

    st.divider()
    st.header("SVR Hyperparameters")

    kernel = st.selectbox("kernel", options=["rbf", "poly", "linear", "sigmoid"], index=0)
    C = st.slider("C", min_value=0.01, max_value=100.0, value=1.0, step=0.01)
    gamma_choice = st.selectbox("gamma", options=["scale", "auto"], index=0)
    epsilon = st.slider("epsilon", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

    st.divider()
    st.header("Predict")

    st.divider()
    st.header("Prediction inputs")
    # For simplicity, if user provides a single feature, ask for that; otherwise ask for first feature only.
    x_input = st.number_input(
        "Value for first feature",
        value=50.0,
        step=1.0,
        help="When multiple features exist, other features are filled with their median from the loaded dataset.",
    )




# Load + validate
if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to parse uploaded CSV. Error: {e}")
        st.stop()
else:
    try:
        data = load_data(csv_path)
    except Exception as e:
        st.error(f"Failed to load CSV at '{csv_path}'. Error: {e}")
        st.stop()

# Resolve target column
if target_col not in data.columns:
    st.error(f"CSV must contain the selected target column: '{target_col}'.")
    st.stop()

feature_cols = [c for c in data.columns if c != target_col]
if len(feature_cols) == 0:
    st.error("CSV must have at least one feature column besides the target column.")
    st.stop()

X = data[feature_cols].values
y = data[target_col].values



left, right = st.columns([1, 1])

with left:
    st.subheader("Training")
    if st.button("Train SVR", type="primary"):
        with st.spinner("Training SVR..."):
            model, mse, r2 = train_svr(
                X,
                y,
                kernel=kernel,
                C=C,
                gamma=gamma_choice,
                epsilon=epsilon,
            )
        st.session_state["model"] = model
        st.session_state["metrics"] = {"mse": float(mse), "r2": float(r2)}

        st.success("Training complete")

    if "metrics" in st.session_state:
        st.markdown("### Evaluation (test split)")
        st.metric("Mean Squared Error (MSE)", f"{st.session_state['metrics']['mse']:.6f}")
        st.metric("R² Score", f"{st.session_state['metrics']['r2']:.6f}")


with right:
    st.subheader("Make a Prediction")
    if "model" not in st.session_state:
        st.info("Train the model first to enable predictions.")
    else:
        model = st.session_state["model"]

        # If there's only one feature, use it directly.
        # Here we assume the first feature column corresponds to input x for the dataset structure.
        if len(feature_cols) == 1:
            input_vec = np.array([[x_input]], dtype=float)
        else:
            # If multiple features exist, we only provide the first for simplicity.
            # Remaining features are set to their training median.
            medians = data[feature_cols].median(numeric_only=True).to_dict()
            payload = []
            for col in feature_cols:
                payload.append(float(x_input) if col == feature_cols[0] else float(medians[col]))
            input_vec = np.array([payload], dtype=float)

        pred = float(model.predict(input_vec)[0])
        st.metric("Predicted x", f"{pred:.6f}")

        with st.expander("Prediction details", expanded=False):
            details = {
                "feature_columns": feature_cols,
                "input_vector": input_vec.flatten().tolist(),
                "kernel": kernel,
                "C": C,
                "gamma": gamma_choice,
                "epsilon": epsilon,
            }
            st.json(details)

