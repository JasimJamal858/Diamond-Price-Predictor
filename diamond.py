import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score

st.title("Diamond Price Prediction")

@st.cache_data
def load_data():
    from seaborn import load_dataset
    data = pd.read_csv('diamond_cleaned.csv')
    return data

data = load_data()
st.write("Dataset Preview")
st.dataframe(data)

target = "price"
columns = data.columns.tolist()
feature_options = [e for e in columns if e != target]

features = st.multiselect("Select Features",feature_options,default=["carat", "depth", "table", "x", "y", "z"])


if "model" not in st.session_state:
    st.session_state.model = None
if "train_columns" not in st.session_state:
    st.session_state.train_columns = None
if "feature_list" not in st.session_state:
    st.session_state.feature_list = None

if st.button("Train Model"):
    if not features:
        st.error("Please select at least one feature!")
    else:
        X = data[features]

        y = data[target]

        X_encoded = pd.get_dummies(X, drop_first=True)

        x_train, x_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

        model = LinearRegression()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success(f" Model trained successfully!")
        st.metric("Mean Absolute Percentage Error", f"{mape:.4f}")
        st.metric("R-Squared Score", f"{r2:.4f}")

        st.write("Sample Predictions vs Actual Prices")
        result_df = pd.DataFrame({
            "Actual Price": y_test.values[:10], 
            "Predicted Price": y_pred[:10]
        })
        st.dataframe(result_df)

        
        st.session_state.model = model
        st.session_state.train_columns = X_encoded.columns.tolist()
        st.session_state.feature_list = features
        st.session_state.r2 = r2
        st.session_state.mape = mape


st.markdown("---")
st.subheader(" Manual Price Prediction")

if st.session_state.model is None:
    st.info("👆 Train the model first using the button above!")
else:
    st.info(f"📊 Using features: {', '.join(st.session_state.feature_list)}")
    
    manual_inputs = {}
    col1, col2 = st.columns(2)
    
    for i, col in enumerate(st.session_state.feature_list):
        if pd.api.types.is_numeric_dtype(data[col]):
            
            min_val, max_val = data[col].min(), data[col].max()
            default_val = float(data[col].median())
            with (col1 if i % 2 == 0 else col2):
                manual_inputs[col] = st.number_input(
                    f"**{col}** (range: {min_val:.1f} - {max_val:.1f})",
                    min_value=min_val * 0.5, max_value=max_val * 1.5,
                    value=default_val, step=0.01
                )
        else:
            #
            categories = sorted(data[col].dropna().unique())[:10]  
            with (col1 if i % 2 == 0 else col2):
                manual_inputs[col] = st.selectbox(
                    f"**{col}**",
                    options=categories,
                    index=0
                )

    if st.button("🎯 Predict Diamond Price", type="primary"):
        #
        new_df = pd.DataFrame([manual_inputs])
        new_df = new_df.fillna(0)  

        
        new_encoded = pd.get_dummies(new_df, drop_first=True)
        
        
        new_encoded = new_encoded.reindex(
            columns=st.session_state.train_columns,
            fill_value=0
        )


        prediction = st.session_state.model.predict(new_encoded)[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Price", f"${prediction:,.2f}")
        with col2:
            st.metric("Training R²", f"{st.session_state.r2:.4f}")
        
        
