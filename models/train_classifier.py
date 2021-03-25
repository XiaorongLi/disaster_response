import sys
import numpy as np
import pandas as pd
import re

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
#import xgboost as xgb 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
import pickle


def load_data(database_filepath):
    '''
    Parameters
    ----------
    database_filepath : string, e.g. 'message.db'

    Returns
    -------
    X: dataframe, messages
    Y: dataframe, labels
    category_names: list
    
    '''
    
    string = 'sqlite:///' + database_filepath
    engine = create_engine(string)    
    table_name = re.split('[.,/]', database_filepath)[-2]
    print(table_name)
    df = pd.read_sql_table(table_name, engine)
    
    #df = pd.read_sql('SELECT * FROM message', con=conn)
    X = df['message']
    Y = df.iloc[:, 4: ]
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    '''Function normalizes case, removes punctuation, stems, lemmatizes and 
    parse a message into separate words.
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens \
              if word not in stop_words]
    
    return tokens


def build_model():
    '''Function builds the pipeline for the final random forest classifier.

    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=10, \
                                max_depth=5, max_features=0.4, max_sample=0.8)))
    ])
    
    parameters = {
    # determine exact parameter names from pipeline.get_params()
    'vect__min_df': [0.005, 0.01],
    'vect__max_df': [0.25, 0.5, 1.0]
    #'clf__estimator__n_estimators': [100, 200],
    #'clf__estimator__max_features': ['sqrt', 'log2'],
    #'clf__estimator__min_samples_leaf': [2,5]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''Function predicts output based on the trained model and Xtest

    Parameters
    ----------
    model : trained model.
    X_test : dataframe
    Y_test : dataframe
    category_names : list

    Returns
    -------
    model report
    '''
    Ypred = model.predict(X_test)
    
    #turn Ypred into a dataframe
    Ypred = pd.DataFrame(Ypred)
    Ypred.columns = category_names
    report = classification_report(Y_test, Ypred, target_names = category_names)
    return report


def save_model(model, model_filepath):
    '''Function saves the trained model into a pickle file'''
    pkl_filename = 'model_filepath'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
    return 0
    
    


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