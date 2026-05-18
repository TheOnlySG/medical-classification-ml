import pandas as pd
import joblib


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

cancer_df = pd.read_csv('data/cancer.csv')

cancer_df.drop(columns=['id' , 'Unnamed: 32']  , inplace=True)

cancer_df['diagnosis']  =  cancer_df['diagnosis'].replace(['M','B'] , [1,0])
cancer_df['diagnosis'] = cancer_df['diagnosis'].astype(int)


x = cancer_df.drop(columns= ['diagnosis'])
y = cancer_df['diagnosis']

scaler = StandardScaler()

x_train , x_test , y_train , y_test = train_test_split(
    x , y , random_state=42 , test_size=0.2
)

x_train = scaler.fit_transform(x_train)

model = LogisticRegression()
model.fit(x_train , y_train)


joblib.dump(model , 'models/cancer_model.pkl')
joblib.dump(scaler , 'models/cancer_scaler.pkl')

print("Model Built Successfully")
