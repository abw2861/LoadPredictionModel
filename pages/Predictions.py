import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os.path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# function to read csv file
def get_data(file, index_col_number):
    df = pd.read_csv(file, index_col=index_col_number, parse_dates=True)
    return df

# function to create and fit model
@st.cache_data
def get_MLP_model(filename, _scaler):
    X_train_scaled = _scaler.transform(X_train)
    
    # model
    MLP_model = MLPRegressor(
        random_state=random_state, 
        hidden_layer_sizes=hidden_layers,
        solver=optimizer,
        learning_rate_init=learning_rate,
        max_iter=epochs, 
        activation=activation_function,
        verbose=True
    )

    MLP_model.fit(X_train_scaled, y_train)

    pickle.dump(MLP_model, open(filename, 'wb'))    # save model

    return MLP_model

# function creating scaler
def get_scaler(train_df):
    # scaling data 0-1
    scaler = MinMaxScaler()
    scaler.fit(train_df)    # scaler fit on training set
    return scaler

# function to evaluate prediction metrics
def evaluate_model(model, _scaler):
    # predict
    X_test_scaled = _scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mape = mean_absolute_percentage_error(y_test, y_pred)
    st.write(f'RMSE: {round(test_rmse,3)} (MW)\n MAPE: {round(test_mape*100,3)}%')

# function to get a prediction using model
def predict(model, _scaler):
    wknd = np.where((datepicker.weekday() >= 5), 1, 0) 
    wh = np.where((wknd == 0) & (timepicker.hour >= 8) & (timepicker.hour <= 17),1,0)
    summer = np.where((datepicker.month >= 5) & (datepicker.month <= 9),1,0)

    input_vals = {'temp':input_temp, 'dew_point':input_dew_point, 'feels_like':input_feels_like, 'is_workhour':wh.item(), 'is_weekend':wknd.item(),  'is_summer':summer.item()}
    userinput = pd.DataFrame.from_dict([input_vals])
    user_pred = model.predict(_scaler.transform(userinput))     # predict on scaled input

    return user_pred

st.set_page_config(page_title='Load Prediction', page_icon='📈')
st.markdown('# Predictions')
st.write('Enter your values below to get a prediction.')

# hyperparameters
random_state = 2
hidden_layers = (128, 64, 32)
activation_function = 'relu'
optimizer = 'adam'
learning_rate = 0.005
epochs = 500
model_filename = 'models/MLP_model.pickle'
dataset_filename = 'datasets/full_data.csv'

# full dataset
full_df = get_data(dataset_filename, 0)

# creating inputs and target
y = full_df['load']
X = full_df[['temp', 'dew_point', 'feels_like', 'is_workhour', 'is_weekend', 'is_summer']]
# split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=random_state)

# create scaler
scaler = get_scaler(X_train)

# load model, if not loaded, create model
if os.path.isfile(model_filename):
    MLPModel = pickle.load(open(model_filename, 'rb'))
else:
    print('Model not found in directory, running new model creation script')
    MLPModel = get_MLP_model(model_filename, scaler)

input_container = st.container()

# input container: getting the inputs and prediction
with input_container:
    input_temp = st.number_input('Temp (F)')
    input_dew_point = st.number_input('Dew Point (F)')
    input_feels_like = st.number_input('Feels Like (F)')
    datepicker = st.date_input('Date')
    timepicker = st.time_input('Time', step=60, help='Tip: Manually enter a time by clicking the time field once and then entering your input. Highlighting the time without selecting the field first will *not* work.')
    st.text(" ")
    button = st.button('Get Prediction')
    if button:
        prediction = predict(MLPModel, scaler)
        st.write('#### Predicted Load (MW):', round(float(prediction),4))

for i in range(3):
    st.text(" ")
st.markdown("""----""")

# show model metrics
expander = st.expander("View model version and accuracy metrics.")
with expander:
    st.write(f'Using model: {model_filename}')
    evaluate_model(MLPModel, scaler)