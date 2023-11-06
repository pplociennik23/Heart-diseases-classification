# Heart Disease classification project
#### Classification of cardiovascular diseases based on Cleveland Heart Disease database

### Table of Contents
* [General info](#General-info)
* [Database info](#Database-info)
* [Technologies](#Technologies)
* [Setup](#Setup)

### General Info

In the described project, a mechanism was implemented to assist in assessing the health status of a patient for the suspicion of heart disease (whether such a disease is present or not). The classifier was built based on the "Cleveland Heart Disease" database provided by the Cleveland Clinic Foundation.

The scope of the project included a series of activities related to data preparation for the classification process. This involved obtaining the database, analyzing the information contained in it, removing irrelevant data, and organizing the remaining content of the database appropriately. Subsequently, by comparing the effectiveness of classification on a selected portion of the dataset, the best algorithm was chosen to build the research model.

Next, the target model was appropriately trained on the training dataset, and classification was performed on the test dataset using this model. Finally, the results of the classification were presented using a series of metrics and charts.

### Database info

The "Cleveland Heart Disease" database contains data from 297 patients, 
who are described by 75 different attributes. Among the numerous features, 14 attributes have been distinguished, 
such as age, gender, resting blood pressure, serum cholesterol, resting ECG, maximum heart rate achieved, exercise-induced angina, and other important parameters, 
which can be the main risk factors for the development of cardiovascular disease.

These attributes are:
* Age
* Gender:
  - 1 = **male**
  - 0 = **female**
* Type of chest pain (cp) including:
  - Value = 1 - **typical angina** 
  - Value = 2 - **atypical angina**
  - Value = 3 - **non-anginal pain**
  - Value = 3 - **asymptomatic pain**
* Resting blood pressure (trestbps) - in mm/Hg at admission to hospital
* Serum cholesterol (chol) in mg/dl
* Fasting sugar > 120 mg/dl (fbs)
  - 1 = **true**
  - 0 = **false**
* Resting electrocardiography results (restecg)
  - Value = 0 - **normal**
  - Value = 1 - **with abnormal ST-T wave** (T wave inversion and/or
  ST segment elevation or depression > 0.05 mV)
  - Value = 2 - **representing probable or definite
  left ventricular hypertrophy according to Estes criteria**
* Maximum heart rate (thalach) achieved
* Exercise-induced angina (exang)
  - 1 = **yes**
  - 0 = **no**
* Exercise-induced ST segment depression relative to rest
(oldpeak)
* Peak ST segment slope during exercise (slope)
  - Value = 1 - **ascending**
  - Value = 2 - **flat**
  - Value = 3 - **falling**
* Number of major vessels (0-3) stained by fluoroscopy (ca)
* Parameter (thal):
  - Value = 3 - **normal**
  - Value = 6 - **defect fixed**
  - Value = 7 - **reversible defect**
* Heart disease diagnosis - angiographic disease status (num)
  - Value = 0 - **no disease**
  - Value = 1 - **disease occurs**

The remaining attributes are personal data of the patients 
that are not directly related to their health status 
(such as address data). Most of these details were encrypted to preserve patient anonymity. 
Therefore, it was necessary to export from the database only those data that are related to the patient's health condition 
(the 14 attributes described above were selected from the database).


### Technologies

**Programming Language**
* Python 3.10 

**Libraries**
* Pandas
* Seaborn
* Matplotlib
* Scikit-learn

### Setup

- Step 1. Download repository 
- Step 2. Set up python interpreter
- Step 3. Download necessary libraries
- Step 4. Run HeartDiseasesClassification.py class

