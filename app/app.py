from flask import Flask , render_template , request
import joblib
import numpy as np
import pandas as pd
import graph_utils as gp


#creating a flask app
app = Flask(__name__)


#models
cancer_model = joblib.load('models/cancer_model.pkl')
cancer_scaler = joblib.load('models/cancer_scaler.pkl')

diabetes_model = joblib.load('models/diabetes_model.pkl')
diabetes_scaler = joblib.load('models/diabetes_scaler.pkl')

heart_model = joblib.load('models/heart_model.pkl')
heart_scaler = joblib.load('models/heart_scaler.pkl')



@app.route('/') #means attach this function to the url route
def home():
    return render_template("landingPage.html")


@app.route('/heart' , methods = ['GET' , 'POST'])
def heart():
    if request.method == 'POST':
        name = request.form['patient_name']
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        input_df = pd.DataFrame([{
                'age': age,
                'sex': sex,
                'cp': cp,
                'trestbps': trestbps,
                'chol': chol,
                'fbs': fbs,
                'restecg': restecg,
                'thalach': thalach,
                'exang': exang,
                'oldpeak': oldpeak,
                'slope': slope,
                'ca': ca,
                'thal': thal
            }])

        scaled_input = heart_scaler.transform(input_df)
        prediction = heart_model.predict(scaled_input)

        if prediction[0] == 1:
            result = ('has a heart disease')
        else :
            result = ("does not have a heart disease")

        gp.heart_chol_graph(chol)
        gp.heart_thalach_graph(thalach)
        gp.heart_oldpeak_graph(oldpeak)

        patient_values = {
            'cp': cp,
            'thalach': thalach,
            'oldpeak': oldpeak,
            'ca': ca,
            'age': age,
            'thal': thal,
            'chol': chol
        }
        gp.heart_final_graph(patient_values)
        comparison_table = gp.heart_comparison_table(patient_values)

        return render_template(
            'heart_report.html',
            prediction = result,
            comparison_table=comparison_table
        )
    

    return render_template("heart.html")

@app.route('/cancer' , methods = ['GET' , 'POST'])
def cancer():


    if request.method == 'POST': #means when the user clicks submit , the backend will recieve a post request so if that happens , collect the data
        name = request.form['patient_name']
        radius_mean = float(request.form['radius_mean'])
        texture_mean = float(request.form['texture_mean'])
        perimeter_mean = float(request.form['perimeter_mean'])
        area_mean = float(request.form['area_mean'])
        concavity_mean = float(request.form['concavity_mean'])
        concave_points_mean = float(request.form['concave_points_mean'])
        radius_worst =  float(request.form['radius_worst'])
        perimeter_worst = float(request.form['perimeter_worst'])
        area_worst = float(request.form['area_worst'])
        concave_points_worst = float(request.form['concave_points_worst'])

        input_df = pd.DataFrame([{
            'radius_mean': radius_mean,
            'texture_mean': texture_mean,
            'perimeter_mean': perimeter_mean,
            'area_mean': area_mean,
            'concavity_mean': concavity_mean,
            'concave points_mean': concave_points_mean,
            'radius_worst': radius_worst,
            'perimeter_worst': perimeter_worst,
            'area_worst': area_worst,
            'concave points_worst': concave_points_worst
        }])

        scaled_input = cancer_scaler.transform(input_df)
        prediction = cancer_model.predict(scaled_input)
        
        if prediction[0] == 1:
            result = 'malignent / has cancer'
        else:
            result = 'Benign / dosent has cancer'

        gp.cancer_radius_worst(radius_worst)
        gp.cancer_concave_worst(concave_points_worst)
        gp.cancer_area_worst(area_worst)
        
        patient_values = {
            'radius_worst': radius_worst,
            'area_worst': area_worst,
            'concave points_worst': concave_points_worst,
            'texture_mean': texture_mean,
            'concave points_mean': concave_points_mean,
            'perimeter_worst': perimeter_worst
        }
        gp.cancer_final(patient_values)
        comparison_table = gp.cancer_comparison_table(patient_values)
            
        return render_template(
            'cancer_report.html',
            prediction = result,
            comparison_table = comparison_table
        )
    return render_template("cancer.html")


@app.route('/diabetes' , methods = ['GET' , 'POST'])
def diabetes():
    if request.method == 'POST':
        name = request.form['patient_name']
        Age = int(request.form['Age'])
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])

        input_df = pd.DataFrame([{
            'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age
        }])

        scaled_input = diabetes_scaler.transform(input_df)
        prediction = diabetes_model.predict(scaled_input)

        if prediction[0] == 1:
            result = "Diabetic"
        else :
            result = "Not Diabetic"

        return render_template(
            'diabetes_report.html',
            prediction = result
        )

    return render_template("diabetes.html")



if __name__ == "__main__":
    app.run(debug=True)