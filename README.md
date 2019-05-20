# Disaster Response Pipeline Project
A Flask web app that can help emergency workers analyze incoming messages and sort them into specific categories to speed up aid and contribute to more efficient distribution of people and other resources.
### Dependencies
*pandas
*numpy
*flask
*sqlalchemy
*plotly
*NLTK
*sklearn
*joblib


### Description
-** ETL pipeline (data directory):**
	*Merges messages and categories datasets
	*Cleans the data and save it into a SQLite database
	
-** Natural Language Processing and Machine Learning Pipeline (models directory):**
	*Split the dataset into training and test sets
	*Build a ML pipeline
	*Train the dataset using GridSearchCV
	*Save the model as a *.pickle file
	
-**Flas Web app (app directory):
	

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
