import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

heart_df = pd.read_csv('data/heart.csv')

x = heart_df.drop(columns= ['target'])
y = heart_df['target']

x_train , x_test , y_train , y_test = train_test_split(x,y , test_size= 0.2 , random_state=42)


scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

model = RandomForestClassifier(n_estimators=100 , random_state=42)
model.fit(x_train , y_train)

joblib.dump(model , 'models/heart_model.pkl')
joblib.dump(scaler , 'models/heart_scaler.pkl')


print("Heart Model Saved Successfully")