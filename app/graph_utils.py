import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

heart_df = pd.read_csv('data/heart.csv')

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
    plt.savefig('app/static/graphs/heart_cholesterol.jpg')
    plt.close()

def heart_thalach_graph(thalach):
    plt.figure(figsize=(10,6))
    patient_thalach = thalach
    patient_category_x = 1.05

    sns.boxenplot(
        data = heart_df,
        x = 'target',
        y = 'thalach',
        palette=['green' , 'red']
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
    plt.savefig('app/static/graphs/heart_thalach.jpg')
    plt.close()


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
    plt.savefig('app/static/graphs/heart_oldpeak.jpg')
    plt.close()


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
    plt.savefig('app/static/graphs/heart_feature_importance.jpg')

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
    healthy_median = healthy_patients[features].median(0)
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