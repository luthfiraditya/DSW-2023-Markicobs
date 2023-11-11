import streamlit as st
import pandas as pd
import pickle

model_low = pickle.load(open('model_low.pkl', 'rb'))
model_mid = pickle.load(open('model_mid.pkl', 'rb'))
model_high = pickle.load(open('model_high.pkl', 'rb'))

columns_low = ['Tenure Months', 'Payment Method']
columns_mid = ['Total Expense (Thou. IDR)', 'Behaviour_Combination']
columns_high = ['Total Expense (Thou. IDR)', 'Behaviour_Combination']

def predict_churn(model, columns, input_data):
    input_df = pd.DataFrame([input_data], columns=columns)

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]

    return prediction[0], probability[0]

def main():
    st.title("Telecom Churn Prediction Markicobss")
    st.write("Enter the customer details below to predict churn.")

    device_type = st.radio("Select your device type:", ['Low', 'Mid', 'High'])

    if device_type == 'Low':
        model = model_low
        columns = columns_low

        tenure = st.slider("Tenure Months", 0, 100, 1)
        payment_method = st.selectbox("Payment Method", [0, 1, 2, 3], format_func=lambda x: ['Credit', 'Debit', 'Digital Wallet', 'Pulsa'][x])
        
        input_data = {
            'Tenure Months': tenure,
            'Payment Method': payment_method
        }

    elif device_type == 'Mid' or device_type == 'High':
        if device_type == 'Mid':
            model = model_mid
            columns = columns_mid
        else:
            model = model_high
            columns = columns_high

        total_expense = st.number_input("Total Expense (Thou. IDR)")
        behavior_combination = st.selectbox("Behaviour Combination", [0, 1, 2], format_func=lambda x: ['Option 1', 'Option 2', 'Option 3'][x])
        
        input_data = {
            'Total Expense (Thou. IDR)': total_expense,
            'Behaviour_Combination': behavior_combination
        }

    churn_prediction, churn_probability = predict_churn(model, columns, input_data)
    
    st.subheader("Churn Prediction")
    if churn_prediction >= 0.4:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is unlikely to churn.")

    st.subheader("Churn Probability")
    st.write("The probability of churn is:", churn_probability)

if __name__ == '__main__':
    main()
