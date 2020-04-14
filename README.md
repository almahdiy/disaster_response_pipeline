# Disaster Response Pipeline Project

### Summary:
This project is an analysis for messages that were sent during disasters. The goal is to create an ETL and a machine learning pipelines to classify the messages into pre-defined categories, to help emergency workers respond to them effectivly. 

### Files:
- app
    - templates
        - go.html
        - master.html
    - run.py
- data
    - disaster_categories.csv
    - disaster_messages.csv
    - process_data.py
- models
    - train_classifier.py

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Attribution:
The data set used in this project was provided by Figure Eight.