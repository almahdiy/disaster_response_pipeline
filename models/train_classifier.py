import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
#from joblib import dump, load
import pickle

import nltk
nltk.download(['punkt', 'wordnet'])




def load_data(database_filepath):
    """
    params:
    - database_filepath: path to the database file where the data is stored.

    returns:
    - X: the input variable, contains the text
    - Y: the target variable, contains all the category columns
    - category_names: list of all the categories a message could have
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("DisasterResponse", con=engine)

    X = df["message"].values

    Y = df[['request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]

    category_names = ['request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']


    return [X, Y, category_names]


def tokenize(text):
    """
    params:
    - text: a string to tokenize

    returns:
    - clean_tokens: a list of tokens generated from the words within the parameter text, 
                    all in lower case & lemmatized
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    for i in range(len(tokens)):
        tokens[i] = lemmatizer.lemmatize(tokens[i]).lower().strip()
    
    return tokens


def build_model():
    """
    Builds the machine learning pipeline.

    returns:
    - cv: a GridSearchCV object
    """
    pipeline = Pipeline([
        ("vectorize", CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()), 
        ("classifier", MultiOutputClassifier(RandomForestClassifier()))
    ])

    #Finding the best model..
    parameters = {
        # 'tfidf__norm': ['l1', 'l2', None],
        # 'tfidf__use_idf': [True, False],
        # 'tfidf__smooth_idf': [True, False],
        # 'tfidf__sublinear_tf': [True, False],
        # 'classifier__estimator__bootstrap': [True, False],
        'classifier__estimator__criterion': ['gini', 'entropy'],
        'classifier__estimator__max_features': ['auto', 'sqrt', 'log2'],
        # 'classifier__estimator__n_estimators': [10, 100],
        # 'classifier__estimator__n_jobs': [-1],
        # 'classifier__estimator__warm_start': [True, False], 
        
    }

    cv = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2, scoring='recall_micro')

    return cv 


  
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Prints the classification report, which includes multiple model evaluation metrics

    params:
    - model: the machine learning model to evaluate
    - X_test: the input features
    - Y_test: the known correct values of Y
    - category_names: names of all the categories
    """

    Y_pred = model.predict(X_test)

    print(classification_report(Y_test, Y_pred, target_names=category_names))



def save_model(model, model_filepath):
    """
    params:
    - model: the model we want to save.
    - model_filepath: the path & file name where we want to save the model.

    """
    pickle.dump(model, open(model_filepath, 'wb'))


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
        
        model = model.best_estimator_

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