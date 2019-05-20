import sys
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM Messages", engine)
    X = df.message
    Y = df.iloc[:, 4:]
    return X,Y,list(Y.columns)
    pass


def tokenize(text):
      # Normalize Text
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    tokens = word_tokenize(text)
    # Remove Stopwords
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    #clean_tokens = []
    #reduce words to their root form
    lemmed = [lemmatizer.lemmatize(w).strip() for w in tokens]
    #lemmatize verbs
    lemmed = [lemmatizer.lemmatize(w,pos='v').strip() for w in lemmed]
    return lemmed


def build_model():
    pipeline = Pipeline([('tfidf',TfidfVectorizer()),
                    ('clf',MultiOutputClassifier(MultinomialNB()))])
    parameters = {
       'tfidf__max_df': (0.25, 0.5, 0.75,1),
       'tfidf__use_idf': (True, False),
       'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
       'clf__estimator__alpha':[1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
    }

    return GridSearchCV(pipeline,param_grid=parameters,n_jobs=-1)
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Category: ", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
    pass


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f: 
        pickle.dump(model, f, -1)      
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()