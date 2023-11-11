import streamlit as st
import pandas as pd
import pickle 
import numpy as np


model_low = pickle.load(open('model_low.pkl', 'rb'))
model_mid = pickle.load(open('model_mid.pkl', 'rb'))
model_high = pickle.load(open('model_high.pkl', 'rb'))

columns_low = ['Tenure Months', 'Payment Method']
columns_mid=['Total Expense (Thou. IDR)', 'Behaviour_Combination']
columns_high=['Total Expense (Thou. IDR)', 'Behaviour_Combination']

df_low = pd.read_csv('lowend_result.csv',delimiter=';')
df_mid = pd.read_csv('midend_result.csv',delimiter=';')
df_high = pd.read_csv('highend_result.csv',delimiter=';')


def get_user_combination(user_input,df_combi):
    # Assuming df_mid is your DataFrame
    # Replace this with the actual DataFrame and column names

    # Condition to filter the DataFrame based on user input
    condition = (
        (df_combi['Games Product'] == user_input['Games Product']) &
        (df_combi['Music Product'] == user_input['Music Product']) &
        (df_combi['Education Product'] == user_input['Education Product']) &
        (df_combi['Call Center'] == user_input['Call Center']) &
        (df_combi['Use MyApp'] == user_input['Use MyApp']) &
        (df_combi['Payment Method'] == user_input['Payment Method']) &
        (df_combi['Video Product'] == user_input['Video Product'])
    )

    # Filter the DataFrame
    filtered_df = df_combi.loc[condition]

    # Check if any rows match the condition
    if not filtered_df.empty:
        # Retrieve the 'Combination' value
        result_combination = filtered_df['Combination'].iloc[0]
        result_combination=int(result_combination.split()[-1])
        return result_combination
    else:
        return "No matching combination"


def predict_churn(model, columns, input_data):
    input_df = pd.DataFrame([input_data], columns=columns)

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    max_probabilities = np.max(probability, axis=1)

    if prediction[0]==0 :
        if 0.5<max_probabilities<0.91:
            churn_risk="Low"
        else:
            churn_risk="least concern"
    else:
        if 0.5<max_probabilities<0.54:
            churn_risk="Notable"
        else:
            churn_risk="Threatening"


    return prediction[0], probability[0], churn_risk

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
        
        Games_Product = st.selectbox("Games Product", ['Yes', 'No'])
        Music_Product = st.selectbox("Music Product", ['Yes', 'No'])
        Education_Product = st.selectbox("Education Product", ['Yes', 'No'])
        Call_Center = st.selectbox("Call Center", ['Yes', 'No'])
        Use_MyApp = st.selectbox("Use MyApp", ['Yes', 'No'])
        Payment_Method = st.selectbox("Payment Method", ['Pulsa', 'Digital Wallet', 'Debit', 'Credit'])
        Video_Product = st.selectbox("Video Product", ['Yes', 'No'])

        user_behavior = {
            'Games Product': Games_Product,
            'Music Product': Music_Product,
            'Education Product': Education_Product,
            'Call Center': Call_Center,
            'Use MyApp': Use_MyApp,
            'Payment Method': Payment_Method,
            'Video Product': Video_Product
        }

        behavior_combination=get_user_combination(user_behavior,df_mid)

        input_data = {
        'Total Expense (Thou. IDR)':total_expense,'Behaviour_Combination':behavior_combination}
       

    churn_prediction, churn_probability,churn_risk = predict_churn(model, columns, input_data)
    
    st.subheader("Churn Prediction")
    if churn_prediction==0:
        st.write("The customer is unlikely to churn.")
    else:
        st.write("The customer is likely to churn.")

    
    max_prob=np.max(churn_probability)*100

    st.subheader("Churn probability")
    st.write("The probability of churn is:", round(max_prob,2))


    st.subheader("Churn Risk")
    st.write("The Churn risk of customer is:", churn_risk)
    
if __name__ == '__main__':
    main()
