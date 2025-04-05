import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS  # Importing the CORS module
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier  # For classification

# Constants
MODEL_FILE = "lottery_model.pkl"
DATA_FILE = "lottery_data.csv"
COLOR_MODEL_FILE = "color_model.pkl"  # File to save the color prediction model
NUMBER_WIN_RECORD = "number_win_loss.csv"
COLOR_WIN_RECORD = "color_win_loss.csv"
color_win_loss_result = "N/A"
number_win_loss_result = "N/A"
color_message_flag = 0
start_telegram_service = False
# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)


# Function to load or initialize the color prediction model
def load_color_model():
    try:
        # Attempt to load an existing model
        model = joblib.load(COLOR_MODEL_FILE)
    except (FileNotFoundError, EOFError):
        # If model doesn't exist, create a new model
        model = SGDClassifier(max_iter=1000, tol=1e-3)
    return model

# Function to prepare data for color prediction
def prepare_color_data(data, n_previous=10):
    X, y = [], []

    # Convert numbers to colors (0 for Red, 1 for Green)
    data['color'] = data['result'].apply(lambda x: 0 if x % 2 == 0 else 1)

    for i in range(len(data) - n_previous):
        X.append(data['color'][i:i + n_previous].values)
        y.append(data['color'][i + n_previous])

    # Ensure X and y are numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Check if data is non-empty
    if X.size == 0 or y.size == 0:
        return None, None

    return X, y

# Function to train and predict colors
def train_and_predict_color():
    # Load the latest lottery data
    data = pd.read_csv(DATA_FILE)

    # Prepare data for training
    X, y = prepare_color_data(data)

    # If there's no valid data, return None
    if X is None or y is None:
        return None, None, len(data)

    # Load the color prediction model
    model = load_color_model()

    # Update the model incrementally
    model.partial_fit(X, y, classes=[0, 1])  # Binary classification

    # Save the updated color model
    joblib.dump(model, COLOR_MODEL_FILE)

    # Get the last 10 colors for prediction
    last_10_colors = data['color'][-10:].values
    prediction = model.predict([last_10_colors])

    return prediction[0], len(data)

# Route to get color predictions
@app.route('/api/color/predict', methods=['GET'])
def predict_color():
    predicted_color, data_size = train_and_predict_color()

    if predicted_color is None:
        return jsonify({"error": "Not enough data to train the color model."}), 400

    # Map predicted color back to its string representation
    color_name = "Red" if predicted_color == 0 else "Green"

    return jsonify({
        "predicted_color": color_name,
        "data_size": data_size
    }), 200
# Function to load or initialize the model
def load_model():
    try:
        # Attempt to load an existing model
        model = joblib.load(MODEL_FILE)
    except (FileNotFoundError, EOFError):
        # If model doesn't exist, create a new model
        model = SGDRegressor(max_iter=1000, tol=1e-3)
    return model

# Function to save lottery data into a CSV file
def save_lottery_data(draw_time, result):
    # Prepare the data to be saved
    new_data = {'drawTime': [draw_time], 'result': [result]}
    new_data_df = pd.DataFrame(new_data)

    # Append to the existing data CSV
    try:
        # Load the existing data from CSV
        data_df = pd.read_csv(DATA_FILE)

        # Check if the new data already exists in the CSV
        if not ((data_df['drawTime'] == draw_time) & (data_df['result'] == result)).any():
            # If no matching entry, append the new data
            data_df = pd.concat([data_df, new_data_df], ignore_index=True)
            data_df.to_csv(DATA_FILE, index=False)
            print("New data saved.")
        else:
            print("Duplicate data detected, not saving.")
    except FileNotFoundError:
        # If the file doesn't exist, create a new one and save the data
        new_data_df.to_csv(DATA_FILE, index=False)
        print("New data saved (new file created).")


# Function to prepare data for model training
def prepare_data(data, n_previous=10):
    X, y = [], []
    for i in range(len(data) - n_previous):
        X.append(data['result'][i:i + n_previous].values)
        y.append(data['result'][i + n_previous])

    # Ensure X and y are numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Check if data is non-empty
    if X.size == 0 or y.size == 0:
        return None, None

    return X, y


# Function to track accuracy (error) of the prediction
def track_accuracy(predicted, actual):
    return abs(predicted - actual)


# Train and predict with the model
def train_and_predict():
    # Load the latest lottery data
    data = pd.read_csv(DATA_FILE)

    # Prepare data for training
    X, y = prepare_data(data)

    # If there's no valid data, return None
    if X is None or y is None:
        return None, None, len(data)

    # Load the model
    model = load_model()

    # Update model incrementally
    model.partial_fit(X, y)

    # Save the updated model
    joblib.dump(model, MODEL_FILE)

    # Get the last 10 results for prediction
    last_10_results = data['result'][-10:].values
    prediction = model.predict([last_10_results])

    # Track the error for the prediction
    actual_result = data['result'].iloc[-1]
    error = track_accuracy(prediction[0], actual_result)

    return prediction[0], error, len(data)


def calculate_predicted_number_range(predicted_number):
    integer_part = int(predicted_number)
    decimal_part = predicted_number - integer_part
    # Logic to determine the range based on decimal part
    if 0.4 <= decimal_part <= 0.6:
        valid_numbers = [integer_part - 1, integer_part, integer_part + 1, integer_part + 2]
    elif decimal_part < 0.4:
        valid_numbers = [integer_part - 1, integer_part, integer_part + 1]
    else:  # decimal_part > 0.6
        valid_numbers = [integer_part, integer_part + 1, integer_part + 2]
    return valid_numbers


def calculate_number_win_loss(predicted_number, actual_number):
    integer_part = int(predicted_number)
    decimal_part = predicted_number - integer_part
    print("entered calculate number win function: PN: ",predicted_number,"AN: ",actual_number)
    # Logic to determine the range based on decimal part
    if 0.4 <= decimal_part <= 0.6:
        valid_numbers = [integer_part - 1, integer_part, integer_part + 1, integer_part + 2]
    elif decimal_part < 0.4:
        valid_numbers = [integer_part - 1, integer_part, integer_part + 1]
    else:  # decimal_part > 0.6
        valid_numbers = [integer_part, integer_part + 1, integer_part + 2]

    # Check if actual number is within the valid range
    return "win" if actual_number in valid_numbers else "lose"


def calculate_color_win_loss(predicted_color, actual_color):
    print("entered calculate color win function: PC: ",predicted_color,"AC: ",actual_color)
    return "win" if predicted_color == actual_color else "lose"


def save_win_loss_record(filename, time, result):
    # Use current system time in ISO 8601 format
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    record = pd.DataFrame({"time": [current_time], "result": [result]})

    try:
        # Append to existing CSV
        data = pd.read_csv(filename)
        data = pd.concat([data, record], ignore_index=True)
    except FileNotFoundError:
        # Create a new file if it doesn't exist
        data = record

    data.to_csv(filename, index=False)


# Global variables to store pending predictions
pending_number_prediction = None
pending_color_prediction = None


def send_msg_telegram(message, flag=333):  # 333 code for prediction request 5 for alert
    bot_token = "7907312021:AAH4Hf9NcgFqKC_wsKZG1lE0cnEgd6SluJw"
    chat_id = "-1001348318693"
    if flag == 5:
        chat_id = "832429046"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message
    }
    response = requests.post(url, data=payload)
    print("message sent to telegram; flag : ",flag)
    return response.json()


def alert_telegram_bot(color_win_loss_result):
    global color_message_flag  # Refer to the global variable

    # Check the result of color window
    if color_win_loss_result == "lose":
        color_message_flag += 1  # Increment flag if result is "LOWS"
    elif color_win_loss_result == "win":
        color_message_flag = 0  # Reset flag if result is "WIN"

    # If flag reaches 4, send the message to Telegram
    if color_message_flag == 5:
        send_msg_telegram("Daddy.....! Its Time", color_message_flag)
        # Reset the flag after sending the message
        color_message_flag = 0
    print("color_msg_flag : ",color_message_flag)
    return color_message_flag


@app.route('/api/data/save', methods=['POST'])
def save_lottery_data_route():
    global pending_number_prediction, pending_color_prediction, color_win_loss_result, number_win_loss_result

    # Parse incoming data
    data = request.get_json()
    print("Received data:", data)  # Log the received data

    draw_time = data.get('drawTime')
    result = data.get('result')  # This is the actual result for the current draw

    if draw_time is not None and result is not None:
        # Save the new data to CSV
        save_lottery_data(draw_time, result)

        # Handle pending predictions comparison
        if pending_number_prediction is not None and pending_color_prediction is not None:
            # Compare the pending predictions with the actual result
            number_win_loss_result = calculate_number_win_loss(pending_number_prediction, result)
            save_win_loss_record(NUMBER_WIN_RECORD, draw_time, number_win_loss_result)

            actual_color = 0 if result % 2 == 0 else 1  # 0 -> Red, 1 -> Green
            color_win_loss_result = calculate_color_win_loss(pending_color_prediction, actual_color)
            save_win_loss_record(COLOR_WIN_RECORD, draw_time, color_win_loss_result)

            alert_telegram_bot(color_win_loss_result)
            # Clear pending predictions after comparison
            pending_number_prediction = None
            pending_color_prediction = None

        # Train models with the current result and predict the next ones
        predicted_result, error, data_size = train_and_predict()
        predicted_color, _ = train_and_predict_color()

        if predicted_result is None or error is None or predicted_color is None:
            return jsonify({"error": "Not enough data to train one or more models."}), 400

        # Save predictions for comparison with the next incoming result
        pending_number_prediction = predicted_result
        pending_color_prediction = predicted_color

        # Map predicted color to its string representation
        color_name = "Red" if predicted_color == 0 else "Green"
        # telegram prediction service MESSAGE Configuration
        if start_telegram_service:
            predicted_number_range=calculate_predicted_number_range(predicted_result)
            predicted_number_range_str = ", ".join(map(str, predicted_number_range))
            message = (
                f"COLOR WIN\n"
                f"Previous draw COLOR: {color_win_loss_result}\n"
                f"Previous draw NUMBER: {number_win_loss_result}\n"
                f"Next draw COLOR: {color_name}\n"
                f"Next Draw NUMBER: {predicted_number_range_str}"
            )
            send_msg_telegram(message)
            # Respond with predictions for the next result
        return jsonify({
            "message": "Data saved and models updated.",
            "predicted_result": predicted_result,
            "predicted_color": color_name,
            "error": error,
            "data_size": data_size,
            "accuracy": 1 - error / predicted_result if predicted_result != 0 else None
        }), 200
    else:
        return jsonify({"error": "Invalid data"}), 400


# Route to get predictions without saving new data
@app.route('/api/predict', methods=['GET'])
def predict():
    predicted_result, error, data_size = train_and_predict()

    if predicted_result is None or error is None:
        return jsonify({"error": "Not enough data to train the model."}), 400

    return jsonify({
        "predicted_result": predicted_result,
        "error": error,
        "data_size": data_size,
        "accuracy": 1 - error / predicted_result if predicted_result != 0 else None
    }), 200


@app.route('/api/telegram', methods=['GET'])
def toggle_prediction_service():
    global start_telegram_service
    start_telegram_service = not start_telegram_service  # Toggle the state
    status = "started" if start_telegram_service else "stopped"
    print(f"Telegram prediction service {status}")
    return f"Telegram service {status}."


if __name__ == "__main__":
    app.run(debug=True)