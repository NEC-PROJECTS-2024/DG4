from flask import Flask, request, render_template
import pandas as pd
import pickle
app = Flask(__name__)
file = open("model.pkl", 'rb')
model = pickle.load(file)
data = pd.read_csv('insurance.csv')
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    sex = request.form.get('sex')
    bmi = float(request.form.get('bmi'))
    children = int(request.form.get('children'))
    smoker = request.form.get('smoker')
    region = request.form.get('region')

    # One-hot encode categorical variables
    sex_encoded = 1 if sex == 'female' else 0
    smoker_encoded = 1 if smoker == 'yes' else 0
    region_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    region_encoded = region_mapping.get(region.lower(), -1)

    if region_encoded == -1:
        return 'Invalid region value'

    # Create a DataFrame with the input features
    input_data = pd.DataFrame([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]],
                              columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

    # Make the prediction
    prediction = model.predict(input_data)[0]
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)
