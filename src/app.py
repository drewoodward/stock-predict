# app.py

from flask import Flask, jsonify
import src.stock_prediction as stock_prediction 

app = Flask(__name__)

# Initialize and train the model on startup
sp500 = stock_prediction.load_sp500_data()
sp500 = stock_prediction.preprocess_data(sp500)
rf_model = stock_prediction.train_rf_model(sp500)

@app.route('/predict/daily', methods=['GET'])
def predict_daily():
    """API endpoint to get a daily stock prediction."""
    prediction, date = stock_prediction.predict_next_day(rf_model, sp500)
    return jsonify({
        "latest_date": date,
        "prediction": prediction  # 1 indicates an expected increase, 0 indicates a decrease
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
