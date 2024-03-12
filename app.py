from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load the saved models
logistic_reg_model = joblib.load('logistic_reg_model.pkl')
knn_model = joblib.load('knn_model.pkl')
naive_bayes_model = joblib.load('naive_bayes_model.pkl')
svm_model = joblib.load('svm_model.pkl')
random_forest_model = joblib.load('random_forest_model.pkl')
decision_tree_model = joblib.load('decision_tree_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = {}
    if request.method == 'POST':
        input_data = request.form['input_data']
        input_data_list = [float(x.strip()) for x in input_data.split(',')]
        
        # Perform predictions using different models
        predictions['logistic_reg'] = 'Malignant' if make_prediction(input_data_list, logistic_reg_model) == 0 else 'Benign'
        predictions['knn'] = 'Malignant' if make_prediction(input_data_list, knn_model) == 0 else 'Benign'
        predictions['naive_bayes'] = 'Malignant' if make_prediction(input_data_list, naive_bayes_model) == 0 else 'Benign'
        predictions['svm'] = 'Malignant' if make_prediction(input_data_list, svm_model) == 0 else 'Benign'
        predictions['random_forest'] = 'Malignant' if make_prediction(input_data_list, random_forest_model) == 0 else 'Benign'
        predictions['decision_tree'] = 'Malignant' if make_prediction(input_data_list, random_forest_model) == 0 else 'Benign'
        print("Predictions:", predictions)
        
        return render_template('index.html', predictions=predictions)
    
    return render_template('index.html')

def make_prediction(input_data, model):
    try:
        # Perform the prediction using the model
        input_data_reshaped = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_data_reshaped)
        print("Prediction:", prediction)
        
        return prediction[0]
    except ValueError:
        # Handle the case where input data cannot be converted to numbers
        print(ValueError)
        return None

if __name__ == '__main__':
    app.run(debug=True)
