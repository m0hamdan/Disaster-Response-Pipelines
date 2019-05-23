# Disaster Response Pipeline Project

### Table of Contents

1. [Dependencies](#depend)
2. [Project Motivation](#motivation)
3. [Content](#files)
4. [Results](#analysis)
5. [Instructions] (#instructions)
6. [Licensing](#licensing)
7. [Acknowledgements] (#ack)


### Dependencies <a name="depend"></a>
1. pandas
2. numpy
3. flask
4. sqlalchemy
5. plotly
6. NLTK
7. sklearn
8. joblib


### Project Motivation:<a name="motivation"></a>

A Flask web app that can help emergency workers analyze incoming messages and sort them into specific categories to speed up aid and contribute to more efficient distribution of people and other resources.

### Content: <a name="files"></a>
-** ETL pipeline (data directory):**
	*Merges messages and categories datasets
	*Cleans the data and save it into a SQLite database
	
-** Natural Language Processing and Machine Learning Pipeline (models directory):**
	*Split the dataset into training and test sets
	*Build a ML pipeline
	*Train the dataset using GridSearchCV
	*Save the model as a *.pickle file
	
-** Flask Web app (app directory):
	

### Instructions:<a name="instructions"></a>

1. Clone the repository: git clone https://github.com/m0hamdan/Disaster-Response-Pipelines.git

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
    - To run ML pipeline that trains classifier and saves
        'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'

3. Run the following command in the app's directory to run your web app.
    'python run.py'

4. Go to http://0.0.0.0:3001/

### Licensing <a name="licensing"></a>


### Acknowledgements <a name="ack"></a>

1. [Figure Eight](https://www.figure-eight.com/) for providing the messages and categories datasets used to train the model
