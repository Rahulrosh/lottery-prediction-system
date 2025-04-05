# 🎯 Lottery Prediction and Analysis System

A Python-based intelligent system for predicting lottery outcomes using machine learning. This project combines classification and regression models with real-time learning, automated performance tracking, and seamless integration with APIs and Telegram bots.

---

## 🧠 Technology Stack

- **Programming Language:** Python  
- **Frameworks:** Flask (API), Telegram Bot API  
- **Machine Learning:** `SGDRegressor`, `SGDClassifier`  
- **Libraries:** Pandas, NumPy, Scikit-learn, FPDF  
- **Deployment Tools:** REST APIs, Telegram Messaging

---

## ⚙️ Features

- 🔢 **Lottery Number Prediction:**  
  Regression-based prediction using past results.

- 🎨 **Color Prediction (Red/Green):**  
  Binary classification model for predicting outcome color.

- 🔁 **Incremental Model Training:**  
  Models are continuously updated with real-time data, improving accuracy over time.

- 📊 **Performance Tracking & Win/Loss Analysis:**  
  Tracks model performance with every draw, storing results and accuracy.

- 📄 **PDF Report Generation:**  
  Auto-generated PDF summaries for prediction accuracy and result distribution.

- 📲 **Telegram Bot Integration:**  
  Instant notifications and prediction alerts via Telegram.

- 🧩 **Chrome Extension Integration:**  
  A custom content script fetches real-time results from the gambling website and sends them to the backend API. It also logs the predictions in the browser’s developer tools for instant monitoring.

- 🔌 **REST API Endpoints:**  
  Easy-to-use endpoints for predictions, data updates, and system integration.

---

## 🚀 Getting Started

###  Clone the Repository

```bash
git clone https://github.com/your-username/lottery-prediction-system.git
cd lottery-prediction-system

 Chrome Extension Setup

The chrome-extension folder inside this project contains all the files needed for the extension.

🧩 To load the extension in Chrome:
	1.	Open Google Chrome
	2.	Go to chrome://extensions/
	3.	Enable Developer Mode (toggle at the top right)
	4.	Click Load unpacked
	5.	Select the chrome-extension folder inside this project
	6.	Navigate to the target lottery website and open Developer Tools (F12) → Console to see real-time logs and data submission events

📬 Telegram Bot Setup
	•	Create a bot using BotFather.
	•	Replace bot_token in your code with your bot’s API key.
	•	Customize alerts and commands as needed.

📌 Notes
	•	This project is experimental and based on statistical patterns; it does not guarantee actual lottery success.
