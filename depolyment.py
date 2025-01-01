import streamlit as st

st.title("AI-Driven Demand Forecasting")

# Upload historical sales data
uploaded_file = st.file_uploader("Upload Sales Data", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", data.head())
    
    # Train and predict
    data.rename(columns={"date": "ds", "sales": "y"}, inplace=True)
    model = Prophet()
    model.fit(data)
    
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    st.write("Forecast Results:", forecast.tail())
    st.line_chart(forecast[['ds', 'yhat']])
