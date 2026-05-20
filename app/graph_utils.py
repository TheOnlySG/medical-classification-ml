import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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