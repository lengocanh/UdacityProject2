import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie

import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals    
    df_categories = df.drop(columns=['id', 'message', 'original', 'genre'])
    categories_name = df_categories.columns.tolist()
    categories_count_1 = df_categories.apply(lambda x: (x == 1).sum()).tolist()
    categories_count_0 = df_categories.apply(lambda x: (x == 0).sum()).tolist()
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    y=categories_name,
                    x=categories_count_1,
                    name='Messages in category',
                    orientation='h',
                    marker=dict(
                        color='rgba(246, 78, 139, 0.6)',
                        line=dict(color='rgba(246, 78, 139, 1.0)', width=1)
                    )
                ),
                Bar(
                    y=categories_name,
                    x=categories_count_0,
                    name='Messages not in category',
                    orientation='h',
                    marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=1)
                    )
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'barmode': 'stack',
                'height': '900',
                'yaxis':{'automargin':True}
                
            }
        }
    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()