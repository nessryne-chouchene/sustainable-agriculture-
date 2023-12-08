
import RPi.GPIO as GPIO
import time
import requests
from requests.exceptions import RequestException



#OUR CODE USES LONG SHORT-TERM MEMORY (LSTM) NEURAL NETWORK
#IT PREDICTS WETHER IT WILL RAIN TOMORROW BASED ON WEATHER FEATURES FROM THE PREVIOUS TWO DAYS
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
#THE DATA
rain = 0
data = {
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15', '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24','2023-01-25'],
    'Max Temperature': [25, 23, 22, 20, 18, 21, 24, 26, 28, 30, 29, 27, 25, 23, 22, 20, 18, 21, 24, 26, 28, 30, 29, 27, 25],
    'Min Temperature': [15, 14, 13, 10, 8, 12, 15, 16, 18, 20, 19, 17, 15, 14, 13, 10, 8, 12, 15, 16, 18, 20, 19, 17, 15],
    'Wind Speed': [10, 8, 12, 15, 14, 9, 11, 13, 16, 18, 17, 14, 12, 10, 9, 11, 1, 14, 16, 13, 10, 9, 1, 12, 14],
    'Rain Tomorrow': [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)
#OUR FEATURES
features = df[['Max Temperature', 'Min Temperature', 'Wind Speed']].values
target = df['Rain Tomorrow'].values
scaler = StandardScaler()
features = scaler.fit_transform(features)
# CREATING SEQUENCES
sequence_length = 2 # Using data from the previous two days
sequences = []
for i in range(len(features) - sequence_length + 1):
    sequences.append(features[i:i+sequence_length, :])

X = np.array(sequences)
y = target[sequence_length-1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
accuracy = model.evaluate(X_test, y_test)[1]
print(f'Test Accuracy: {accuracy*100:.2f}%')

#PREDICTION FOR TOMORROW BASED ON THE TWO PREVIOUS DAYS
new_sequence = np.array([[26, 16, 15], [26, 16, 14]]) #THESE VALUES ARE MEANT TO BE TAKEN FROM ACTUAL SENSORS

if new_sequence.shape[1] != features.shape[1]:
    print(f"Number of features in the new sequence ({new_sequence.shape[1]}) "
          f"should match the number of features in the training data ({features.shape[1]}).")
else:
    new_sequence_scaled = scaler.transform(new_sequence)
    new_sequence_reshaped = new_sequence_scaled.reshape(1, -1, features.shape[1])
    prediction = model.predict(new_sequence_reshaped)
    prediction_binary = (prediction > 0.5)
    # RESULT!!
    if prediction_binary:
        rain = 1
    else:
        rain = 0


# Constants for the GPIO pins and weather API
SALINITY_PIN = 4
MOISTURE_PIN = 17
TEMPERATURE_PIN = 27
SWEET_WATER_VALVE_PIN = 18
SALTY_WATER_VALVE_PIN = 22


def read_sensor(pin):
    pass

GPIO.setmode(GPIO.BCM)

# Set up the GPIO pins
GPIO.setup(SALINITY_PIN, GPIO.IN)
GPIO.setup(MOISTURE_PIN, GPIO.IN)
GPIO.setup(TEMPERATURE_PIN, GPIO.IN)
GPIO.setup(SWEET_WATER_VALVE_PIN, GPIO.OUT)
GPIO.setup(SALTY_WATER_VALVE_PIN, GPIO.OUT)

while True:
    salinity = read_sensor(SALINITY_PIN)
    moisture = read_sensor(MOISTURE_PIN)
    temperature = read_sensor(TEMPERATURE_PIN)
    should_make_sweet_water=False
    should_make_salty_water=False
    irrigation = False
    
    # Code to determine if sweet water or salty water should be made
    # based on the salinity, moisture, temperature, and rain chance

    if moisture < 50:
        irrigation = True
    else:
        irrigation = False

    if irrigation:
        if salinity > 0.5 :
            should_make_sweet_water = True
        elif 0.3 <= salinity <= 0.5:
            if rain==1:
                should_make_salty_water = True
            else:
                should_make_sweet_water = True
        else:
            should_make_salty_water = True

            if should_make_sweet_water:
                GPIO.output(SWEET_WATER_VALVE_PIN, GPIO.HIGH)
                GPIO.output(SALTY_WATER_VALVE_PIN, GPIO.LOW)
            if should_make_salty_water:
                GPIO.output(SWEET_WATER_VALVE_PIN, GPIO.LOW)
                GPIO.output(SALTY_WATER_VALVE_PIN, GPIO.HIGH)

            else:
                GPIO.output(SWEET_WATER_VALVE_PIN, GPIO.LOW)
                GPIO.output(SALTY_WATER_VALVE_PIN, GPIO.LOW)
    time.sleep(300) # Check every 5 minute





