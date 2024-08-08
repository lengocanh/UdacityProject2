### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

This is the python scripts that can be run with Python versions 3.*<br>
Makesure you have following packages installed:
1. numpy
2. pandas
3. sqlalchemy
4. re
5. nltk
6. sklearn

## Project Motivation<a name="motivation"></a>

For this project, I use NLP pipeline and ML pipeline to build the model to detect the category of a message.
The categories regards to the disaster. There are 36 categories:<br>
0                    related
1                    request
2                      offer
3                aid_related
4               medical_help
5           medical_products
6          search_and_rescue
7                   security
8                   military
9                child_alone
10                     water
11                      food
12                   shelter
13                  clothing
14                     money
15            missing_people
16                  refugees
17                     death
18                 other_aid
19    infrastructure_related
20                 transport
21                 buildings
22               electricity
23                     tools
24                 hospitals
25                     shops
26               aid_centers
27      other_infrastructure
28           weather_related
29                    floods
30                     storm
31                      fire
32                earthquake
33                      cold
34             other_weather
35             direct_report

One message can be assign to 1 or multiple categories

## File Descriptions <a name="files"></a>

#### process_data.py
To run ETL pipeline that cleans data and stores in database
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
#### train_classifier.py 
To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
#### run.py
1. Run the following command in the app's directory to run your web app.
    `python run.py`

2. Go to http://0.0.0.0:3001/

## Results<a name="results"></a>

There are some data in categories has 'related-2', after review some data, I suggest it's not related.
And I convert all 'related-2' to 'related-0'<br>

Some data has url in the message, I replace url by "urlplaceholder" in tokenize function<br>

Almost of categories are imbalanced, I used the parameter 'clf__estimator__class_weight': ['balanced', 'balanced_subsample']<br>

I want to try more parameters and features, but the training in my computer take very long time, then I only try with 3 parameters:
1. 'clf__estimator__n_estimators': [20, 40, 60],
2. 'clf__estimator__class_weight': ['balanced', 'balanced_subsample']
3. 'clf__estimator__min_samples_split': [2, 3, 4]

Modify the build_model function to lower or remove parameters to get the model train faster

re.sub is not pickle function, then cannot set the n_jobs param

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

disaster_messages and disaster_categories.csv files are provided by Udacity <br>
master.html and go.html are provided by Udacity<br>
Otherwise, feel free to use the code here as you would like! <br>
