#getting base
FROM python:3.14

#setting the /app folder as workdir as the code is in disease-predictor/app/app.py , and other things
#models , researchnotebooks , train , .gitignore , Dockerfile , README.md are in /
WORKDIR /app

#layer to install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

#copy the whole code
COPY . .


#final layer to run the app
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120", "app.app:app"]


