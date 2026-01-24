import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# ------------------------------------
# 1️⃣ Load & Prepare Data
# ------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Assignment_9/dataset.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.drop("customerID", axis=1, inplace=True)
    df.dropna(inplace=True)

    # Encode categorical features
    le_dict = {}
    for col in df.select_dtypes(include="object"):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict

df, le_dict = load_data()

# ------------------------------------
# 2️⃣ Train Model (on-the-fly)
# ------------------------------------
@st.cache_data
def train_model(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    return model, scaler, X.columns

model, scaler, feature_names = train_model(df)

# ------------------------------------
# 3️⃣ Streamlit Layout
# ------------------------------------
st.title("💼 Telco Customer Churn Prediction")
st.markdown("Predict if a customer will churn based on their details.")

# Sidebar for inputs
st.sidebar.header("Customer Input Features")
input_data = {}

for feature in feature_names:
    if feature in le_dict:  # Categorical column
        # Decode categories for display
        le = le_dict[feature]
        categories = le.classes_
        choice = st.sidebar.selectbox(f"{feature}", options=categories)
        input_data[feature] = le.transform([choice])[0]
    else:  # Numeric column
        col_dtype = df[feature].dtype
        if pd.api.types.is_integer_dtype(col_dtype):
            value = st.sidebar.number_input(
                f"{feature}",
                int(df[feature].min()),
                int(df[feature].max()),
                int(df[feature].median()),
                step=1
            )
        else:
            value = st.sidebar.number_input(
                f"{feature}",
                float(df[feature].min()),
                float(df[feature].max()),
                float(df[feature].median()),
                step=0.1
            )
        input_data[feature] = value

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    st.write(f"**Prediction:** {'Churn 🚨' if prediction == 1 else 'Stay 🙂'}")
    st.write(f"**Probability of Churn:** {proba*100:.2f}%")

# ------------------------------------
# 4️⃣ Plots
# ------------------------------------
st.subheader("Churn Distribution")
fig, ax = plt.subplots()
df["Churn"].value_counts().plot(kind="bar", ax=ax)
ax.set_xlabel("Churn (0 = No, 1 = Yes)")
ax.set_ylabel("Number of Customers")
st.pyplot(fig)

st.subheader("Tenure vs Churn")
fig2, ax2 = plt.subplots()
df.boxplot(column="tenure", by="Churn", ax=ax2)
ax2.set_xlabel("Churn")
ax2.set_ylabel("Tenure (Months)")
ax2.set_title("Tenure vs Churn")
st.pyplot(fig2)

st.subheader("Monthly Charges vs Churn")
fig3, ax3 = plt.subplots()
df.boxplot(column="MonthlyCharges", by="Churn", ax=ax3)
ax3.set_xlabel("Churn")
ax3.set_ylabel("Monthly Charges")
ax3.set_title("Monthly Charges vs Churn")
st.pyplot(fig3)

