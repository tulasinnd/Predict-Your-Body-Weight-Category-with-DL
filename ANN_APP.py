import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import streamlit as st

@st.cache_resource()
def load_model_and_scaler():
    # load the dataset
    df=pd.read_csv(r'dataset/weight.csv')
    dele=['FCVC','NCP','FAF','TUE']
    cat=['Gender','fhistory','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS' ]
    cont=['Age','Height','Weight','CH2O']
    df = df.drop(dele, axis=1)

    # Create dummy variables for categorical columns
    data = pd.get_dummies(df, columns=cat)
    
    # dependent and independent variable split
    y = data.iloc[:, 4]

    skip_columns = [4]  # Index of 4th column to skip
    all_columns = np.arange(data.shape[1])  # Indexes of all columns
    keep_columns = np.delete(all_columns, skip_columns)  # Indexes of columns to keep

    X = data.iloc[:, keep_columns].copy()

    # encode the target variable using LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # scale the input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # define the ANN model
    model = Sequential()
    model.add(Dense(32, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    # compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    return scaler, model, le

#___________________________________USER INPUT FORM FOR PREDICTION_____________________________________

# Questions
st.title("Predict Your Body Weight Category with DL")
st.markdown("### Personal Information")
D_gender = st.selectbox("1 What is your gender? 👧👦", ["Female", "Male"])
st.write('')
st.write('')
age = st.slider("2 What is your age? 🎂", 0, 100,25,key=1)
st.write('')
st.write('')
height = st.number_input("3 What is your height (in meters)? 📏", value=1.5, step=0.01)
st.write('')
st.write('')
weight = st.number_input("4 What is your weight (in kilograms)? ⚖️", value=50, step=1)
st.write('')
st.write('')
st.markdown("### Lifestyle Habits")    
D_family_history = st.selectbox("5 Has a family member suffered or suffers from overweight? 🧬", ["yes", "no"])
st.write('')
st.write('')
D_FAVC = st.selectbox("6 Do you eat high caloric food frequently? 🍔", ["yes", "no"])
st.write('')
st.write('')
D_CAEC = st.selectbox("7 Do you eat any food between meals? 🍫", ["no", "Sometimes", "Frequently", "Always"])
st.write('')
st.write('')
D_SMOKE = st.selectbox("8 Do you smoke? 🚬", ["yes", "no"])
st.write('')
st.write('')
st.write('9 How much water do you drink daily? 💧')
c1,c2,c3=st.columns([2,5,3])
with c1:
    st.write('Less than a liter')
with c2:
    st.write("<p style='text-align: center;'>Between 1 and 2 L</p>", unsafe_allow_html=True)
with c3:
    st.write('More than 2 L')
ch2o=st.slider("", 1.0, 3.0, step=0.01,key=2)
st.write('')
st.write('')
D_SCC = st.selectbox("10 Do you monitor the calories you eat daily? 🔢", ["yes", "no"])
st.write('')
st.write('')
D_CALC = st.selectbox("11 How often do you drink alcohol? 🍻🍷🥂", ["no", "Sometimes", "Frequently", "Always"])
st.write('')
st.write('')
D_MTRANS = st.selectbox("12 Which transportation do you usually use? 🚗🏍️🚲🚃🚶‍♂️", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])


# USER INPUT FOR PREDICTING 
l=[D_gender,age,height,weight,D_family_history,D_FAVC,D_CAEC,D_SMOKE,ch2o,D_SCC,D_CALC,D_MTRANS]
Gender=D_gender
Age=age
Height=height
Weight=weight
CH2O=ch2o
Gender_Female, Gender_Male = (1, 0) if D_gender == 'Female' else (0, 1)
fhistory_yes, fhistory_no = (1, 0) if D_family_history == 'yes' else (0, 1)
FAVC_no, FAVC_yes = (0, 1) if D_FAVC == 'yes' else (1, 0)
SMOKE_yes, SMOKE_no = (1, 0) if D_SMOKE == 'yes' else (0, 1)
# define the dictionary for CAEC values
CAEC_dict = {'Always': [1,0,0,0], 'Frequently': [0,1,0,0], 'Sometimes': [0,0,1,0], 'no': [0,0,0,1]}
# use the dictionary to assign the values
CAEC_Always, CAEC_Frequently, CAEC_Sometimes, CAEC_no = CAEC_dict[D_CAEC]
SCC_yes, SCC_no = (1, 0) if D_SCC == 'yes' else (0, 1)
CALC_dict = {'Always': [1,0,0,0], 'Frequently': [0,1,0,0], 'Sometimes': [0,0,1,0], 'no': [0,0,0,1]}
# use the dictionary to assign the values
CALC_Always, CALC_Frequently, CALC_Sometimes, CALC_no = CALC_dict[D_CAEC]
MTRANS_Automobile, MTRANS_Bike, MTRANS_Motorbike, MTRANS_Public_Transportation, MTRANS_Walking = (
(1,0,0,0,0) if D_MTRANS == 'Automobile' else
(0,1,0,0,0) if D_MTRANS == 'Bike' else
(0,0,1,0,0) if D_MTRANS == 'Motorbike' else
(0,0,0,1,0) if D_MTRANS == 'Public_Transportation' else
(0,0,0,0,1) if D_MTRANS == 'Walking' else
(0,0,0,0,0)
)

# USER INPUT AS LIST
feature_list = [Age, Height, Weight, CH2O, Gender_Female, Gender_Male, fhistory_no, fhistory_yes, 
                FAVC_no, FAVC_yes, CAEC_Always, CAEC_Frequently, CAEC_Sometimes, CAEC_no, SMOKE_no, 
                SMOKE_yes, SCC_no, SCC_yes, CALC_Always, CALC_Frequently, CALC_Sometimes, CALC_no, 
                MTRANS_Automobile, MTRANS_Bike, MTRANS_Motorbike, MTRANS_Public_Transportation, MTRANS_Walking]

#_________________________________________________________________________________________________


# PREDICTION
if st.button("Predict"):
        
    # Call the function to load the model and scaler
    scaler, model, le= load_model_and_scaler()
    

    X_new = np.array([feature_list])

    # scale the input values
    X_new_scaled = scaler.transform(X_new)

    # make a prediction using the trained model
    y_pred = model.predict(X_new_scaled)

    # decode the predicted category using LabelEncoder
    y_pred_decoded = le.inverse_transform(np.argmax(y_pred, axis=1))

    st.write('Predicted category:')

    weight_category = y_pred_decoded[0]  # Replace this with the actual weight category

    if weight_category == 'Normal_Weight':
        with st.container():
            st.write("<span style='color:green;font-weight:bold;font-size:72px;'>Normal weight</span>", unsafe_allow_html=True)
            st.balloons()
    elif weight_category == 'Overweight_Level_I':
        with st.container():
            st.write("<span style='color:orange;font-weight:bold;font-size:72px;'>Overweight level I</span>", unsafe_allow_html=True)
    elif weight_category == 'Overweight_Level_II':
        with st.container():
            st.write("<span style='color:orange;font-weight:bold;font-size:72px;'>Overweight level II</span>", unsafe_allow_html=True)
    elif weight_category == 'Obesity_Type_I':
        with st.container():
            st.write("<span style='color:red;font-weight:bold;font-size:72px;'>Obesity type I</span>", unsafe_allow_html=True)
    elif weight_category == 'Insufficient_Weight':
        with st.container():
            st.write("<span style='color:blue;font-weight:bold;font-size:72px;'>Insufficient weight</span>", unsafe_allow_html=True)
    elif weight_category == 'Obesity_Type_II':
        with st.container():
            st.write("<span style='color:red;font-weight:bold;font-size:72px;'>Obesity type II</span>", unsafe_allow_html=True)
    elif weight_category == 'Obesity_Type_III':
        with st.container():
            st.write("<span style='color:red;font-weight:bold;font-size:72px;'>Obesity type III</span>", unsafe_allow_html=True)
