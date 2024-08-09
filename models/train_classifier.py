import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import joblib

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem  import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

def load_data(database_filepath):
    """Load data from sqlite to X, Y and categories names

    Args:
    database_filepath: string. path to sqlite file

    Returns:
    X: DataFrame with 1 column message
    Y: DataFrate with 36 colums of categories
    category_names: list contain 36 category names
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_response', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = list(Y.columns)
    return X, Y, category_names



def tokenize(text):
    """parse text in to tokens and lemmatize tokens

    Args:
    text: string. the text need to be tokenize

    Returns:
    clean_tokens: list of clean tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """Build the model pipeline to clasify token

    Args:

    Returns:
    cv: GridSearchCV with parameters
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # specify parameters for grid search
    parameters = {
        'clf__estimator__n_estimators': [20, 40, 60],
        'clf__estimator__class_weight': ['balanced', 'balanced_subsample'],
        #'clf__estimator__min_samples_split': [2, 3, 4]
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4)
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """Report the f1 score, precision and recall for each output category of the dataset. 

    Args:
    model: GridSearchCV model after trained
    X_test: DataFrame of messages used to test the model
    Y_test: DataFrame of categories used to test the model
    category_names: list contain 36 category names

    Returns:
    """
    Y_pred = model.predict(X_test)
    for cat in category_names:
        print("Classification Report for category " + cat)
        print(classification_report(Y_test[cat], Y_pred[:, category_names.index(cat)]))



def save_model(model, model_filepath):
    """Save the trained model in to a file 

    Args:
    model: GridSearchCV model after trained
    model_filepath: stirng. File path to save the model

    Returns:
    """
    joblib.dump(model, model_filepath)


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