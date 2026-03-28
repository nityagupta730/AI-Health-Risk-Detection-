from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Model load karo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        blood_pressure = int(request.form['blood_pressure'])
        cholesterol = int(request.form['cholesterol'])
        heart_rate = int(request.form['heart_rate'])

        input_data = np.array([[age, blood_pressure, cholesterol, heart_rate]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        if prediction == 0:
            result = "Normal"
            risk_level = "low"
            message = "Aapka dil bilkul theek hai!"
            percentage = round(probability[0] * 100, 2)
        else:
            result = "At Risk"
            risk_level = "high"
            message = "Please doctor se milein jaldi!"
            percentage = round(probability[1] * 100, 2)

        return render_template('result.html',
                             result=result,
                             risk_level=risk_level,
                             message=message,
                             percentage=percentage,
                             age=age,
                             bp=blood_pressure,
                             chol=cholesterol,
                             hr=heart_rate)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
    