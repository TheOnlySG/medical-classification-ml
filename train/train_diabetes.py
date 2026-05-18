import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

diabetes_df = pd.read_csv('data/diabetes.csv')

x = diabetes_df.drop(columns=['Outcome'])
y = diabetes_df['Outcome']


x_train , x_test , y_train , y_test = train_test_split(
    x , y , random_state=42 ,test_size=0.2
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

model = RandomForestClassifier(n_estimators=200 , random_state=42)
model.fit(x_train , y_train)


joblib.dump(model , 'models/diabetes_model.pkl')
joblib.dump(scaler , 'models/diabetes_scaler.pkl')

print("Model Saved Successfully")
