from flask import Flask, render_template, request, make_response, jsonify
import pickle
import io
import csv
from io import StringIO
import pandas as pd



app = Flask(__name__)
model = pickle.load(open('Diabetes_Prediction.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])

        prediction=model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness,Insulin, BMI, DiabetesPedigreeFunction, Age]])
        if prediction==0:
            return render_template('index.html',prediction_text="You are free from diabetes")
        else:
            return render_template('index.html',prediction_text="You have diabetes")
    else:
        return render_template('index.html')

@app.route('/defaults',methods=['POST'])

def defaults():

    return render_template('index.html')

@app.route('/default',methods=["POST"])

def default():

    return render_template('layout.html')

def transform(text_file_contents):

    return text_file_contents.replace("=", ",")



@app.route('/transform', methods=["POST"])

def transform_view():

    f = request.files['data_file']

    if not f:

        return "No file"



    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)

    csv_input = csv.reader(stream)

    #print("file contents: ", file_contents)

    #print(type(file_contents))

    print(csv_input)

    for row in csv_input:

        print(row)



    stream.seek(0)

    result = transform(stream.read())



    df = pd.read_csv(StringIO(result))

    



    loaded_model = pickle.load(open('Diabetes_Prediction.pkl', 'rb'))

    df['Outcome'] = loaded_model.predict(df)



    #df = df.insert(10, 'Amount_Paid', df['prediction'])



    response = make_response(df.to_csv())

    response.headers["Content-Disposition"] = "attachment; filename=result.csv"

    #response.headers["Content-Type"] = "text/csv"

    return response




if __name__=="__main__":
    app.run(debug=True)