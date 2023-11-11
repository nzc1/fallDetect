from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np 
from scipy.stats import kurtosis, skew
import csv
import datetime
import time

app = Flask(__name__)

# Load trained model
lrmodel, lrscaler = joblib.load('fall_detection_lrmodel.joblib')
rfModel, rfScaler = joblib.load('fall_detection_rfmodel.joblib')

# root
@app.route('/', methods=['GET'])
def home():
    try:        
        return jsonify({'message': 'AfterLook', 'value':True})
                
    except Exception as e:
        return jsonify({'error': str(e)})

# preprocess
@app.route('/process', methods=['POST'])
def preProcess():
    try:
        # Get the JSON data from the request body
        resData = request.json.get('data', [])
        resData1 = request.json.get('gyData', [])
        
        # Extract the arrays
        resArray = [[item['timestamp'], item['x'], item['y'], item['z']] for item in resData]
        resArray1 = [[item['x'], item['y'], item['z']] for item in resData1]
        
        combined_data = list(zip(resArray, resArray1))
        # Flatten the subarrays into a list of lists
        flattened_data = [sublist1 + sublist2 for sublist1, sublist2 in combined_data]
      
        data = np.array(flattened_data)

        # Calculate time starting from 0
        timestampsList = data[:, 0].astype(float)
        start_time = timestampsList[0]
        timestamps_seconds = (timestampsList - start_time) / 1000
        data[:, 0] = timestamps_seconds
        
        # separate data
        accelerometer_x = data[:, 1].astype(float)
        accelerometer_y = data[:, 2].astype(float)
        accelerometer_z = data[:, 3].astype(float)
        gyroscope_x = data[:, 4].astype(float)
        gyroscope_y = data[:, 5].astype(float)
        gyroscope_z = data[:, 6].astype(float)
        timestamps = data[:, 0].astype(float)

        # Define the time window for the 4th and 6th seconds
        window_start_4th = 4.0
        window_end_4th = 5.0
        window_start_6th = 6.0
        window_end_6th = 7.0

        # Extract data within the 4th second window
        acc_x_4th = accelerometer_x[(timestamps >= window_start_4th) & (timestamps < window_end_4th)]
        acc_y_4th = accelerometer_y[(timestamps >= window_start_4th) & (timestamps < window_end_4th)]
        acc_z_4th = accelerometer_z[(timestamps >= window_start_4th) & (timestamps < window_end_4th)]
        gyro_x_4th = gyroscope_x[(timestamps >= window_start_4th) & (timestamps < window_end_4th)]
        gyro_y_4th = gyroscope_y[(timestamps >= window_start_4th) & (timestamps < window_end_4th)]
        gyro_z_4th = gyroscope_z[(timestamps >= window_start_4th) & (timestamps < window_end_4th)]

        # Extract data within the 6th second window
        acc_x_6th = accelerometer_x[(timestamps >= window_start_6th) & (timestamps < window_end_6th)]
        acc_y_6th = accelerometer_y[(timestamps >= window_start_6th) & (timestamps < window_end_6th)]
        acc_z_6th = accelerometer_z[(timestamps >= window_start_6th) & (timestamps < window_end_6th)]
        gyro_x_6th = gyroscope_x[(timestamps >= window_start_6th) & (timestamps < window_end_6th)]
        gyro_y_6th = gyroscope_y[(timestamps >= window_start_6th) & (timestamps < window_end_6th)]
        gyro_z_6th = gyroscope_z[(timestamps >= window_start_6th) & (timestamps < window_end_6th)]

        # Compute the features

        # 1. Maximum acceleration magnitude of the 4th second window
        acc_magnitude_4th = np.sqrt(acc_x_4th**2 + acc_y_4th**2 + acc_z_4th**2)
        acc_max_4th = np.max(acc_magnitude_4th)

        # 2. Maximum gyroscope magnitude of the 4th second window
        gyro_magnitude_4th = np.sqrt(gyro_x_4th**2 + gyro_y_4th**2 + gyro_z_4th**2)
        gyro_max_4th = np.max(gyro_magnitude_4th)

        # 3. Acceleration kurtosis of the whole window
        acc_magnitude = np.sqrt(accelerometer_x**2 + accelerometer_y**2 + accelerometer_z**2)
        acc_kurtosis = kurtosis(acc_magnitude)

        # 4. Gyroscope kurtosis of the whole window
        gyro_magnitude = np.sqrt(gyroscope_x**2 + gyroscope_y**2 + gyroscope_z**2)
        gyro_kurtosis = kurtosis(gyro_magnitude)

        # 5. Maximum linear acceleration of the 4th second window
        linear_acceleration_4th = np.sqrt((acc_x_4th - 0)**2 + (acc_y_4th - 0)**2 + (acc_z_4th - 9.81)**2)
        lin_max_4th = np.max(linear_acceleration_4th)

        # 6. Acceleration skewness of the whole window
        acc_skewness = skew(acc_magnitude)

        # 7. Gyroscope skewness of the whole window
        gyro_skewness = skew(gyro_magnitude)

        # 8. Maximum gyroscope magnitude of the 6th second window
        gyro_magnitude_6th = np.sqrt(gyro_x_6th**2 + gyro_y_6th**2 + gyro_z_6th**2)
        post_gyro_max_6th = np.max(gyro_magnitude_6th)

        # 9. Maximum linear acceleration of the 6th second window
        linear_acceleration_6th = np.sqrt((acc_x_6th - 0)**2 + (acc_y_6th - 0)**2 + (acc_z_6th - 9.81)**2)
        post_lin_max_6th = np.max(linear_acceleration_6th)

        # print calculated features
        calculated_features = [acc_max_4th, gyro_max_4th, acc_kurtosis, gyro_kurtosis, lin_max_4th, acc_skewness, gyro_skewness, post_gyro_max_6th, post_lin_max_6th]
        print(calculated_features)
        print("acc_max_4th:", acc_max_4th)
        print("gyro_max_4th:", gyro_max_4th)
        print("acc_kurtosis:", acc_kurtosis)
        print("gyro_kurtosis:", gyro_kurtosis)
        print("lin_max_4th:", lin_max_4th)
        print("acc_skewness:", acc_skewness)
        print("gyro_skewness:", gyro_skewness)
        print("post_gyro_max_6th:", post_gyro_max_6th)
        print("post_lin_max_6th:", post_lin_max_6th)
                
        return jsonify({'message': 'Preprocessed data', 'features': calculated_features})
    except Exception as e:
        return jsonify({'error': str(e)})

# Predict
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request body
        resData = request.json.get('data', [])
        resData1 = request.json.get('gyData', [])
        
        # Extract the arrays
        resArray = [[item['timestamp'], item['x'], item['y'], item['z']] for item in resData]
        resArray1 = [[item['x'], item['y'], item['z']] for item in resData1]
        
        combined_data = list(zip(resArray, resArray1))
        # Flatten the subarrays into a list of lists
        flattened_data = [sublist1 + sublist2 for sublist1, sublist2 in combined_data]
      
        data = np.array(flattened_data)

        # Calculate time starting from 0
        timestampsList = data[:, 0].astype(float)
        start_time = timestampsList[0]
        timestamps_seconds = (timestampsList - start_time) / 1000
        data[:, 0] = timestamps_seconds
        
        # separate data
        accelerometer_x = data[:, 1].astype(float)
        accelerometer_y = data[:, 2].astype(float)
        accelerometer_z = data[:, 3].astype(float)
        gyroscope_x = data[:, 4].astype(float)
        gyroscope_y = data[:, 5].astype(float)
        gyroscope_z = data[:, 6].astype(float)
        timestamps = data[:, 0].astype(float)

        # Define the time window for the 4th and 6th seconds
        window_start_4th = 4.0
        window_end_4th = 5.0
        window_start_6th = 6.0
        window_end_6th = 7.0

        # Extract data within the 4th second window
        acc_x_4th = accelerometer_x[(timestamps >= window_start_4th) & (timestamps < window_end_4th)]
        acc_y_4th = accelerometer_y[(timestamps >= window_start_4th) & (timestamps < window_end_4th)]
        acc_z_4th = accelerometer_z[(timestamps >= window_start_4th) & (timestamps < window_end_4th)]
        gyro_x_4th = gyroscope_x[(timestamps >= window_start_4th) & (timestamps < window_end_4th)]
        gyro_y_4th = gyroscope_y[(timestamps >= window_start_4th) & (timestamps < window_end_4th)]
        gyro_z_4th = gyroscope_z[(timestamps >= window_start_4th) & (timestamps < window_end_4th)]

        # Extract data within the 6th second window
        acc_x_6th = accelerometer_x[(timestamps >= window_start_6th) & (timestamps < window_end_6th)]
        acc_y_6th = accelerometer_y[(timestamps >= window_start_6th) & (timestamps < window_end_6th)]
        acc_z_6th = accelerometer_z[(timestamps >= window_start_6th) & (timestamps < window_end_6th)]
        gyro_x_6th = gyroscope_x[(timestamps >= window_start_6th) & (timestamps < window_end_6th)]
        gyro_y_6th = gyroscope_y[(timestamps >= window_start_6th) & (timestamps < window_end_6th)]
        gyro_z_6th = gyroscope_z[(timestamps >= window_start_6th) & (timestamps < window_end_6th)]

        # Compute the features

        # 1. Maximum acceleration magnitude of the 4th second window
        acc_magnitude_4th = np.sqrt(acc_x_4th**2 + acc_y_4th**2 + acc_z_4th**2)
        acc_max_4th = np.max(acc_magnitude_4th)

        # 2. Maximum gyroscope magnitude of the 4th second window
        gyro_magnitude_4th = np.sqrt(gyro_x_4th**2 + gyro_y_4th**2 + gyro_z_4th**2)
        gyro_max_4th = np.max(gyro_magnitude_4th)

        # 3. Acceleration kurtosis of the whole window
        acc_magnitude = np.sqrt(accelerometer_x**2 + accelerometer_y**2 + accelerometer_z**2)
        acc_kurtosis = kurtosis(acc_magnitude)

        # 4. Gyroscope kurtosis of the whole window
        gyro_magnitude = np.sqrt(gyroscope_x**2 + gyroscope_y**2 + gyroscope_z**2)
        gyro_kurtosis = kurtosis(gyro_magnitude)

        # 5. Maximum linear acceleration of the 4th second window
        linear_acceleration_4th = np.sqrt((acc_x_4th - 0)**2 + (acc_y_4th - 0)**2 + (acc_z_4th - 9.81)**2)
        lin_max_4th = np.max(linear_acceleration_4th)

        # 6. Acceleration skewness of the whole window
        acc_skewness = skew(acc_magnitude)

        # 7. Gyroscope skewness of the whole window
        gyro_skewness = skew(gyro_magnitude)

        # 8. Maximum gyroscope magnitude of the 6th second window
        gyro_magnitude_6th = np.sqrt(gyro_x_6th**2 + gyro_y_6th**2 + gyro_z_6th**2)
        post_gyro_max_6th = np.max(gyro_magnitude_6th)

        # 9. Maximum linear acceleration of the 6th second window
        linear_acceleration_6th = np.sqrt((acc_x_6th - 0)**2 + (acc_y_6th - 0)**2 + (acc_z_6th - 9.81)**2)
        post_lin_max_6th = np.max(linear_acceleration_6th)

        # calculated features
        features = [acc_max_4th, gyro_max_4th, acc_kurtosis, gyro_kurtosis, lin_max_4th, acc_skewness, gyro_skewness, post_gyro_max_6th, post_lin_max_6th]

        data_point = pd.DataFrame({
            'acc_max': [features[0]],
            # not having a higher impact
            # 'gyro_max': [features[1]],
            'acc_kurtosis': [features[2]],
            'gyro_kurtosis': [features[3]],
            'lin_max': [features[4]],
            'acc_skewness': [features[5]],
            'gyro_skewness': [features[6]],
            'post_gyro_max': [features[7]],
            'post_lin_max': [features[8]]
        })

        # Perform data scaling if necessary (assuming 'scaler' is defined)
        data_point_scaled = rfScaler.transform(data_point)
        
        # Make predictions using loaded model
        prediction = rfModel.predict(data_point_scaled)
        
        message = "Not a fall" if prediction[0] == 0 else "Fall"
        value = False if prediction[0] == 0 else True

        # Return the predictions as JSON
        return jsonify({'message': message, 'isFalled': value})

    except Exception as e:
        return jsonify({'error': str(e)})

# test
@app.route('/test', methods=['POST'])
def testAcc():
    try:
        # Get the JSON data from the request body
        reData = request.json.get('data', [])
        data = [[item['timestamp'], item['x'], item['y'], item['z'], np.sqrt(item['x']**2 + item['y']**2 + item['z']**2)] for item in reData]
        
        # Create a CSV file and write the data to it
        # with open('data1.csv', 'w', newline='') as csv_file:
        #     csv_writer = csv.writer(csv_file)
        #     csv_writer.writerow(['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
        #     for row in flattened_data:
        #         csv_writer.writerow(row)

        time_values = []
        state_values = []
        acc = []

        count = 0
        state = 0

        for line in data:
            # Rest to simulate real-time data
            time.sleep(0.0039)

            count += 1

            time_values.append(float(line[0]))
            acc.append(float(line[4]))

            if count % 200 == 0:
                print('round', state)
                # Compute standard deviation of the last 200 data points
                std = np.std(acc[-200:])

                if std < 0.75 and (state == 1 or state == 2):
                    state = 0
                    print('User at rest. State', 0)

                elif std > 0.75 and (std < 10 and state == 0 or state == 2):
                    state = 1
                    print('User walking. State', 1)

                elif std > 5 and (state == 1 or state == 0):
                    state = 2

                    print('')
                    print('Fall detected at time:', datetime.datetime.fromtimestamp(time_values[-1] / 1000))
                    print('Standard deviation:', np.round(std, 2))
                    print('')
                    # Return the predictions as JSON
                    return jsonify({'message': 'Fall Detected', 'time': datetime.datetime.fromtimestamp(time_values[-1] / 1000)})

                state_values.append(state)
                
        return jsonify({'message': 'Read Completed', 'value': True})
    except Exception as e:
        return jsonify({'error': str(e)})

# Define a route for prediction using Logistic Regression
@app.route('/predict/lr', methods=['POST'])
def predictLR():
    try:
        # Get the JSON data from the request body
        data = request.json

        # Assuming your JSON data contains the features you want to predict
        features = data['features']

        data_point = pd.DataFrame({
            'acc_max': [features[0]],
            'gyro_max': [features[1]],
            'acc_kurtosis': [features[2]],
            'gyro_kurtosis': [features[3]],
            'lin_max': [features[4]],
            'acc_skewness': [features[5]],
            'gyro_skewness': [features[6]],
            'post_gyro_max': [features[7]],
            'post_lin_max': [features[8]]
        })

        # Perform data scaling if necessary (assuming 'scaler' is defined)
        data_point_scaled = lrscaler.transform(data_point)
        
        # Make predictions using your loaded model
        predictions = lrmodel.predict(data_point_scaled)
        
        # Iterate through predictions and build a new list
        prediction_values = []
        for prediction in predictions:
            message = "Not a fall" if prediction == 0 else "Fall"
            # Append each prediction value to the list
            prediction_values.append({'message': message, 'isFalled': prediction.tolist()})

        # Return the predictions as JSON
        return jsonify(prediction_values)

    except Exception as e:
        return jsonify({'error': str(e)})

# Define a route for prediction using Random Forest
@app.route('/predict/rf', methods=['POST'])
def predictRf():
    try:
        # Get the JSON data from the request body
        data = request.json

        # Assuming your JSON data contains the features you want to predict
        features = data['features']
        
        data_point = pd.DataFrame({
            'acc_max': [features[0]],
            # 'gyro_max': [features[1]],
            'acc_kurtosis': [features[1]],
            'gyro_kurtosis': [features[2]],
            'lin_max': [features[3]],
            'acc_skewness': [features[4]],
            'gyro_skewness': [features[5]],
            'post_gyro_max': [features[6]],
            'post_lin_max': [features[7]]
        })

        # Perform data scaling if necessary (assuming 'scaler' is defined)
        data_point_scaled = rfScaler.transform(data_point)
        
        # Make predictions using your loaded model
        predictions = rfModel.predict(data_point_scaled)

        # Iterate through predictions and build a new list
        prediction_values = []
        for prediction in predictions:
            message = "Not a fall" if prediction == 0 else "Fall"
            # Append each prediction value to the list
            prediction_values.append({'message': message, 'isFalled': prediction.tolist()})

        # Return the predictions as JSON
        return jsonify(prediction_values)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)