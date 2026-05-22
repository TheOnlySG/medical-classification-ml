import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

#production handling for multiple images genration at same time
import uuid
import os
import glob
import time

def cleanup_old_graphs():
    current_time = time.time()
    files = glob.glob('app/static/graphs/*')
    for file in files:
        file_age = current_time - os.path.getmtime(file)

        if file_age > 300:
            os.remove(file)
#basically , whenever the file gets 5 min older , we will delete it. 
'''
why have i added this though ? as if 2 users generate the graph at same time , and one of them
refreshes his site , then that will probably cause an issue. this function will handle it perfectly
move to line68 , and you will understand the unique name genration for graphs.
'''



heart_df = pd.read_csv('data/heart.csv')
cancer_df = pd.read_csv('data/cancer.csv')
diabetes_df = pd.read_csv('data/diabetes.csv')

def heart_chol_graph(chol):
    patient_chol = chol
    plt.figure(figsize=(10,6))
    sns.kdeplot(
        data = heart_df[heart_df['target'] == 0],
        x = 'chol',
        fill = True,
        color = 'green',
        label = 'healthy'
    )
    sns.kdeplot(
        data = heart_df[heart_df['target'] == 1],
        x = 'chol',
        fill = True,
        color = 'red',
        label = 'Heart Disease'
    )

    plt.axvline(
        patient_chol,
        color = 'blue',
        linestyle = '--',
        linewidth = 3,
        label = 'Patient'
    )

    plt.title('Cholesterol Distribution Analysis')
    plt.xlabel('cholesterol')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    
    unique_id = uuid.uuid4().hex
    filename = f'heart_cholestrol_{unique_id}.jpg'
    filepath = f'app/static/graphs/{filename}'
    plt.savefig(filepath)
    # plt.savefig('app/static/graphs/heart_cholesterol.jpg') replacing this in every function now

    plt.close()
    return filename


def heart_thalach_graph(thalach):
    plt.figure(figsize=(10,6))
    patient_thalach = thalach
    patient_category_x = 1.05

    sns.boxenplot(
        data = heart_df,
        x = 'target',
        y = 'thalach',
        color = 'silver'
    )

    plt.scatter(
        patient_category_x,
        patient_thalach,
        color = 'blue',
        marker = 'x',
        label = 'patient',
        s = 200
    )
    plt.title('Maximum heartrate analysis')
    plt.xticks([0,1] , ['healthy' , 'heart disease'])
    plt.ylabel('maximum heart rate')
    plt.legend()
    plt.grid(alpha = 0.3)
    plt.tight_layout()

    unique_id = uuid.uuid4().hex
    filename = f'heart_thalach_{unique_id}.jpg'
    filepath = f'app/static/graphs/{filename}'
    plt.savefig(filepath)
    plt.close()
    return filename


def heart_oldpeak_graph(oldpeak):
    patient_oldpeak = oldpeak
    plt.figure(figsize=(10,6))
    sns.kdeplot(
        data = heart_df[heart_df['target'] == 0],
        x = 'oldpeak',
        fill = True,
        color = 'green',
        label = 'healthy'
    )

    sns.kdeplot(
        data = heart_df[heart_df['target'] == 1],
        x = 'oldpeak',
        fill = True,
        color= 'red',
        label = 'Heart Disease'
    )
    plt.axvline(
        patient_oldpeak,
        color = 'blue',
        linestyle = '--',
        linewidth = 3,
        label = 'Patient'
    )

    plt.title('Oldpeak Distribution Analysis')
    plt.xlabel('Oldpeak')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    
    unique_id = uuid.uuid4().hex
    filename = f'heart_oldpeak_{unique_id}.jpg'
    filepath = f'app/static/graphs/{filename}'
    plt.savefig(filepath)
    plt.close()
    return filename


def heart_final_graph(patient_values):
    features = [
        'cp',
        'thalach',
        'oldpeak',
        'ca',
        'age',
        'thal',
        'chol'
    ]

    feature_display_names = [
        'Chest Pain Type',
        'Maximum Heart Rate',
        'ST Depression (Oldpeak)',
        'Major Vessels',
        'Age',
        'Thalassemia',
        'Cholesterol'
    ]

    importances = [
        0.127280,
        0.116200,
        0.113342,
        0.112550,
        0.105167,
        0.095484,
        0.079107
    ]
    patient_df = pd.DataFrame([patient_values])
    scaler = MinMaxScaler()
    scaler.fit(heart_df[features])
    normalized_patient = scaler.transform(patient_df).flatten()
    normalized_patient = np.clip(normalized_patient, 0, 1)


    importance_scaler = MinMaxScaler()
    normalized_importances = importance_scaler.fit_transform(
        np.array(importances).reshape(-1,1)
    ).flatten()
    plt.figure(figsize = (10,6))
    sns.barplot(
        x = normalized_importances,
        y = feature_display_names,
        color = 'gray',
        alpha = 0.8,
        label = 'Model Importance'
    )
    plt.scatter(
        normalized_patient,
        feature_display_names,
        color = 'blue',
        marker = 'x',
        s = 200,
        label = 'Patient Feature Strength'
    )

    plt.xlabel('Patient Feature Strength')
    plt.title('Model Feature Importance vs Patient Profile')
    plt.legend()
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    unique_id = uuid.uuid4().hex

    filename = f'heart_final_{unique_id}.jpg'
    filepath = f'app/static/graphs/{filename}'
    plt.savefig(filepath)
    plt.close()
    return filename

def heart_comparison_table(patient_values):
    features = [
        'cp',
        'thalach',
        'oldpeak',
        'ca',
        'age',
        'thal',
        'chol'
    ]

    feature_display_names = [
        'Chest Pain Type',
        'Maximum Heart Rate',
        'ST Depression (Oldpeak)',
        'Major Vessels',
        'Age',
        'Thalassemia',
        'Cholesterol'
    ]

    healthy_patients = heart_df[heart_df['target'] == 0]
    healthy_median = healthy_patients[features].median()
    patient_values_list = [
        patient_values[feature]
        for feature in features
    ]
    comparison_table = pd.DataFrame({
        'Features' : feature_display_names,
        'Healthy Median' : healthy_median.values ,
        'Your Values': patient_values_list
    })
    comparison_table = comparison_table.round(2)
    return comparison_table.to_dict(orient='records')


def cancer_radius_worst(patient_radius_worst):
    plt.figure(figsize=(10,6))
    sns.kdeplot(
        data = cancer_df[cancer_df['diagnosis'] == 'B'],
        x = 'radius_worst',
        fill =True,
        color = 'green',
        label = 'Benign'
    )

    sns.kdeplot(
        data = cancer_df[cancer_df['diagnosis'] == 'M'],
        x = 'radius_worst',
        fill =True,
        color = 'red',
        label = 'Malignant'
    )
    plt.axvline(
        patient_radius_worst,
        color = 'blue',
        linestyle = '--',
        linewidth=3,
        label = 'Patient'
    )

    plt.title('Radius Worst Distribution Analysis')
    plt.xlabel('Radius Worst')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    
    unique_id = uuid.uuid4().hex
    filename = f'cancer_radius_worst_{unique_id}.jpg'
    filepath = f'app/static/graphs/{filename}'
    plt.savefig(filepath)
    plt.close()
    return filename


def cancer_concave_worst(patient_concave_worst):
    plt.figure(figsize=(10,6))
    sns.kdeplot(
        data = cancer_df[cancer_df['diagnosis'] == 'B'],
        x = 'concave points_worst',
        color = 'green',
        fill = True,
        label = 'Benign'
    )

    sns.kdeplot(
        data = cancer_df[cancer_df['diagnosis'] == "M"],
        x = 'concave points_worst',
        color ='red',
        fill = True,
        label = 'Malignant'
    )

    plt.axvline(
        patient_concave_worst,
        color = 'blue',
        linestyle = '--',
        linewidth = 3,
        label = 'Patient'
    )

    plt.title('Concave Patient Worst vs Distribution Analysis')
    plt.xlabel('Concave Points Worst')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    unique_id = uuid.uuid4().hex
    filename = f'cancer_concave_worst_{unique_id}.jpg'
    filepath = f'app/static/graphs/{filename}'
    plt.savefig(filepath)
    plt.close()
    return filename

def cancer_area_worst(patient_area_worst):
    plt.figure(figsize=(10,6))
    sns.violinplot(
        data = cancer_df[cancer_df['diagnosis'] == 'M'],
        y = 'area_worst',
        color = 'gray',
        inner = 'quartile'
    )
    plt.scatter(
        0.1,
        patient_area_worst,
        color = 'cyan',
        marker = 'x',
        s = 200,
        label = 'Patient'
    )

    plt.title('Malignant Area Worst Distribution')
    plt.ylabel('Area Worst')
    plt.legend()
    plt.grid(alpha = 0.9)
    plt.tight_layout()
    unique_id = uuid.uuid4().hex
    filename = f'cancer_area_worst_{unique_id}.jpg'
    filepath = f'app/static/graphs/{filename}'
    plt.savefig(filepath)
    plt.close()
    return filename


def cancer_final(patient_values):
    features = [
        'radius_worst',
        'area_worst',
        'concave points_worst',
        'texture_mean',
        'concave points_mean',
        'perimeter_worst'
    ]
    feature_display_names = [
        'Radius Worst',
        'Area Worst',
        'Concave Points Worst',
        'Texture Mean',
        'Concave Points Mean',
        'Perimeter Worst'
    ]

    coefficients = [
        1.746043,
        1.734130,
        1.717279,
        1.270087,
        1.226007,
        1.139706
    ]

    patient_df = pd.DataFrame([patient_values])
    patient_scaler = MinMaxScaler()
    patient_scaler.fit(cancer_df[features])
    normalized_patients = patient_scaler.transform(patient_df).flatten()
    normalized_patients = np.clip(normalized_patients,0,1)

    importance_scaler = MinMaxScaler()
    normalized_coeff = importance_scaler.fit_transform(
        np.array(coefficients).reshape(-1,1)
    ).flatten()


    plt.figure(figsize=(10,6))
    sns.barplot(
        x = normalized_coeff,
        y = feature_display_names,
        color = 'gray',
        alpha = 0.8,
        label = 'Model Importance'
    )

    plt.scatter(
        normalized_patients,
        feature_display_names,
        color = 'blue',
        marker = 'x',
        s = 200,
        label = 'Patient Feature Strength'
    )

    plt.xlim(0,1.05) #set limit to graph bars
    plt.xlabel('Relative Importance and Patient Strength')
    plt.title('Cancer AI Attention vs Patient Clinical Profile')
    plt.legend()
    plt.grid(alpha  = 0.3)
    plt.tight_layout()
    unique_id = uuid.uuid4().hex
    filename = f'cancer_final_{unique_id}.jpg'
    filepath = f'app/static/graphs/{filename}'
    plt.savefig(filepath)
    plt.close()
    return filename

def cancer_comparison_table(patient_values):

    features = [
        'radius_worst',
        'area_worst',
        'concave points_worst',
        'texture_mean',
        'concave points_mean',
        'perimeter_worst'
    ]

    feature_display_names = [
        'Radius Worst',
        'Area Worst',
        'Concave Points Worst',
        'Texture Mean',
        'Concave Points Mean',
        'Perimeter Worst'
    ]

    healthy_patients = cancer_df[cancer_df['diagnosis'] == 'B']

    healthy_median = healthy_patients[features].median()

    patient_values_list = [
        patient_values[feature]
        for feature in features
    ]

    comparison_table = pd.DataFrame({
        'Features': feature_display_names,
        'Healthy Median': healthy_median.values,
        'Your Values': patient_values_list
    })

    comparison_table = comparison_table.round(2)

    return comparison_table.to_dict(orient='records')
    plt.close()

def diabetes_glucose(patient_glucose):
    plt.figure(figsize=(10,6))
    sns.kdeplot(
        data = diabetes_df[diabetes_df['Outcome'] == 1],
        x = 'Glucose',
        color = 'red',
        fill = True,
        label = 'Diabetic'
    )
    sns.kdeplot(
        data = diabetes_df[diabetes_df['Outcome'] == 0 ],
        x = 'Glucose',
        color = 'green',
        label = 'Healthy',
        fill = True
    )
    plt.axvline(
        patient_glucose,
        linestyle='--',
        linewidth=3,
        label='Patient',
        color = 'blue'
    )
    plt.title('Glucose Distribution Analysis')
    plt.xlabel('Glucose')
    plt.ylabel('Density')
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    unique_id = uuid.uuid4().hex
    filename = f'diabetes_glucose_{unique_id}.jpg'
    filepath = f'app/static/graphs/{filename}'
    plt.savefig(filepath)
    plt.close()
    return filename

def diabetes_bmi(patient_bmi):
    plt.figure(figsize=(10,6))
    sns.kdeplot(
        data = diabetes_df[diabetes_df['Outcome'] == 0],
        x = 'BMI',
        fill = True,
        color = 'green',
        label = 'Healthy'
    )
    sns.kdeplot(
        data = diabetes_df[diabetes_df['Outcome'] == 1],
        x = 'BMI',
        fill = True,
        color = 'red',
        label = 'Diabetic'
    )
    plt.axvline(
        patient_bmi,
        color = 'blue',
        linestyle = '--',
        linewidth = 3,
        label = 'Patient'
    )
    plt.title('BMI Distribution Analysis')
    plt.xlabel('BMI')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    unique_id = uuid.uuid4().hex
    filename = f'diabetes_bmi_{unique_id}.jpg'
    filepath = f'app/static/graphs/{filename}'
    plt.savefig(filepath)
    plt.close()
    return filename

def diabetes_insulin(patient_insulin):
    plt.figure(figsize=(10,6))
    sns.violinplot(
        data = diabetes_df[diabetes_df['Outcome'] == 1],
        y = 'Insulin',
        color = 'gray',
        inner = 'quartile'
    )

    plt.scatter(
        0,
        patient_insulin,
        color = 'blue',
        marker = 'x',
        s = 200,
        label = 'Patient'
    )
    plt.title('Diabetic Insulin Distribution')
    plt.ylabel('Insulin')
    plt.xticks([])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    unique_id = uuid.uuid4().hex
    filename = f'diabetes_insulin_{unique_id}.jpg'
    filepath = f'app/static/graphs/{filename}'
    plt.savefig(filepath)
    plt.close()
    return filename

def diabetes_main(patient_values):
    features = [
        'Glucose',
        'BMI',
        'Age',
         'DiabetesPedigreeFunction',
        'Insulin',
        'BloodPressure'
    ]
    feature_display_names = [
        'Glucose',
        'BMI',
        'Age',
        'Diabetes Pedigree Function',
        'Insulin',
        'Blood Pressure'
    ]
    importances = [
        0.256770,
        0.165276,
        0.139238,
        0.119710,
        0.092145,
        0.082466
    ]
    patient_df = pd.DataFrame([patient_values])
    patient_Scaler = MinMaxScaler()
    patient_Scaler.fit(diabetes_df[features])
    normalized_patients  = patient_Scaler.transform(patient_df).flatten()
    normalized_patients=np.clip(normalized_patients , 0 , 1)

    important_scaler = MinMaxScaler()
    normalized_coefficients = important_scaler.fit_transform(
        np.array(importances).reshape(-1,1)
    ).flatten()

    plt.figure(figsize=(10,6))

    sns.barplot(
        x = normalized_coefficients,
        y = feature_display_names,
        color = 'gray',
        alpha = 0.8,
        label = 'Model Importance'
    )

    plt.scatter(
        normalized_patients,
        feature_display_names,
        color = 'blue',
        marker = 'x',
        s = 200,
        label = 'Patient Feature Strength'
    )

    plt.xlim(0,1.05)
    plt.xlabel('Relative Importance and Patient Strength')
    plt.title('Diabetes Model Attention vs Patient Clinical Profile')
    plt.legend(fontsize = 13)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    unique_id = uuid.uuid4().hex
    filename = f'diabetes_main_{unique_id}.jpg'
    filepath = f'app/static/graphs/{filename}'
    plt.savefig(filepath)
    plt.close()
    return filename

def diabetes_comparison_table(patient_values):

    features = [
        'Glucose',
        'BMI',
        'Age',
        'DiabetesPedigreeFunction',
        'Insulin',
        'BloodPressure'
    ]

    feature_display_names = [
        'Glucose',
        'BMI',
        'Age',
        'Diabetes Pedigree Function',
        'Insulin',
        'Blood Pressure'
    ]

    healthy_patients = diabetes_df[diabetes_df['Outcome'] == 0]
    healthy_median = healthy_patients[features].median()
    patient_values_list = [
        patient_values[feature]
        for feature in features
    ]
    comparison_table = pd.DataFrame({
        'Features': feature_display_names,
        'Healthy Median': healthy_median.values,
        'Your Values': patient_values_list
    })
    comparison_table = comparison_table.round(2)
    return comparison_table.to_dict(orient='records')





    



