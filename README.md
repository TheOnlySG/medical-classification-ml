<p align="center">
  <img src="assets/github_banner.png" width="100%">
</p>

---

SymptoSense is an ML-powered diagnostic platform that analyzes clinical inputs for Heart Disease, Diabetes, and Tumor prediction while generating explainable, medical-style reports with visual insights and patient comparisons.

Launch Live Deployment - https://symptosense-68xi.onrender.com/ </br>
<i>Hosted on Render free tier — cold starts after inactivity may take <b>30–60 seconds</b>.</i>

---

## Features

<table>
<tr>
<td width="50%">

### Heart Disease Analysis
Analyze cardiovascular health indicators with ML-powered risk prediction and visual comparisons.

</td>

<td width="50%">

### Diabetes Prediction
Evaluate glucose, BMI, insulin, and related metrics through explainable patient analysis.

</td>
</tr>

<tr>
<td width="50%">

### Tumor Diagnostics
Generate interpretable tumor analysis reports using clinical measurement distributions.

</td>

<td width="50%">

### Explainable Reports
Medical-style printable reports with feature importance graphs and patient comparisons.

</td>
</tr>

</table>

---

## Sample Reports

<table width="100%">

<tr>

<td align="center" width="33%">

<a href="./assets/heart_report.pdf">

<b>Heart Disease Report</b>

</a>

</td>

<td align="center" width="33%">

<a href="./assets/diabetes_report.pdf">

<b>Diabetes Report</b>

</a>

</td>

<td align="center" width="33%">

<a href="./assets/cancer_report.pdf">

<b>Tumor Analysis Report</b>

</a>

</td>

</tr>

</table>

---

## How To Use

<table width="100%">

<tr>
<th width="18%">Step</th>
<th>Description</th>
</tr>

<tr>
<td align="center"><b>1</b></td>
<td>
Choose one of the available diagnostic systems:
Heart Disease, Diabetes, or Tumor Analysis.
</td>
</tr>

<tr>
<td align="center"><b>2</b></td>
<td>
Fill in the required patient health parameters using the provided medical input forms.
</td>
</tr>

<tr>
<td align="center"><b>3</b></td>
<td>
Submit the form to allow the ML model to process the patient profile and generate predictions.
</td>
</tr>

<tr>
<td align="center"><b>4</b></td>
<td>
Analyze generated graphs, feature importance visualizations, and patient comparison metrics.
</td>
</tr>

<tr>
<td align="center"><b>5</b></td>
<td>
Export the generated medical-style report as a printable PDF document.
</td>
</tr>

</table>

---

## Tech Stack

<p align="center">

<img src="https://skillicons.dev/icons?i=flask,docker" />

</p>

<p align="center">

<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge">
<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">

</p>

---

## ML Models Used

<table width="100%">

<tr>
<th width="30%">Diagnostic Module</th>
<th width="35%">Model</th>
<th width="35%">Purpose</th>
</tr>

<tr>
<td><b>Heart Disease Analysis</b></td>
<td>Random Forest Classifier</td>
<td>Predicts cardiovascular disease risk using patient clinical indicators.</td>
</tr>

<tr>
<td><b>Diabetes Prediction</b></td>
<td>Random Forest Classifier</td>
<td>Analyzes diabetic health patterns and glucose-related measurements.</td>
</tr>

<tr>
<td><b>Tumor Diagnostics</b></td>
<td>Logistic Regression</td>
<td>Classifies tumors as benign or malignant using clinical measurements.</td>
</tr>

</table>

---

## Local Setup

<table width="100%">

<tr>
<th width="22%">Setup</th>
<th>Description</th>
</tr>

<tr>

<td>

### Local Environment (venv)

</td>

<td>

#### 1. Clone Repository

```bash
git clone https://github.com/TheOnlySG/SymptoSense.git
cd SymptoSense
```

#### 2. Create Virtual Environment

```bash
python -m venv venv
```

##### Linux / macOS

```bash
source venv/bin/activate
```

##### Windows

```bash
venv\Scripts\activate
```

#### 3. Install Requirements

```bash
pip install -r requirements.txt
```

#### 4. Run Using Gunicorn

```bash
gunicorn app.app:app
```

Application will start on:

```bash
http://localhost:8000
```

</td>

</tr>

<tr>

<td>

### Docker Deployment

</td>

<td>

#### 1. Build Docker Image

```bash
docker build -t symptosense .
```

#### 2. Run Docker Container

```bash
docker run -p 5000:8000 symptosense
```

Application will start on:

```bash
http://localhost:5000
```

</td>

</tr>

</table>

---

## License

This project is licensed under the MIT License.
