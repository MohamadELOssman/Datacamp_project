## In-Hospital Mortality Prediction with Machine Learning

This repository contains code for data preprocessing and building machine learning models to predict **in-hospital mortality** for patients admitted to a hospital. 

### Dataset Description

This project assumes you will be using a dataset containing **patient information** relevant to their hospital stay, including:

* **Demographics:** Age, gender, etc.
* **Vital signs:** Heart rate, blood pressure, etc.
* **Laboratory results:** Blood tests, urine tests, etc.
* **Comorbidities:** Existing medical conditions like diabetes, heart disease, etc.
* **Hospital admission details:** Reason for admission, ICU stay, length of stay, etc.
* **Target variable:** **Binary label indicating whether the patient died during hospitalization (1) or survived (0).**

### Work Done

A preprocessing of the data and training of different models is done in the notebook file. 2 submissions corresponding to the machine learning models in the notebook can also be execited with ramp with the command: 
``` 
ramp-test --submission "name of the submission folder" 
```
