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
4. sklearn
5. nltk
6. plotly
7. flask

## Project Motivation<a name="motivation"></a>

For this project, I build the web application which can detect disaster categories of a message that user input.<br>
There are 36 categories:<br>
1	related	<br>
2	request	<br>
3	offer	<br>
4	aid_related	<br>
5	medical_help	<br>
6	medical_products	<br>
7	search_and_rescue	<br>
8	security	<br>
9	military	<br>
10	child_alone	<br>
11	water	<br>
12	food	<br>
13	shelter	<br>
14	clothing	<br>
15	money	<br>
16	missing_people	<br>
17	refugees	<br>
18	death	<br>
19	other_aid	<br>
20	infrastructure_related	<br>
21	transport	<br>
22	buildings	<br>
23	electricity	<br>
24	tools	<br>
25	hospitals	<br>
26	shops	<br>
27	aid_centers	<br>
28	other_infrastructure	<br>
29	weather_related	<br>
30	floods	<br>
31	storm	<br>
32	fire	<br>
33	earthquake	<br>
34	cold	<br>
35	other_weather	<br>
36	direct_report	<br>

To build this application I have implemented following components:
1. A ETL pipeline reads raw data from messages and categories csv files, clean and save data to sqlite
2. A ML pipeline clasifies message. The ML is trained using data from sqlite in step 1
3. A Web application receives message from user and detect its disaster categories. The Web application uses the ML model from step 2.


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

#### master.html
The html template for home page when user access the web application

#### go.html
The html template to display message categories after users type in their message and click on "Clasify Message"

## Results<a name="results"></a>


There are some data in categories has 'related-2', after review some data, I suggest it's not related.
And I convert all 'related-2' to 'related-0'<br>

Some data has url in the message, I replace url by "urlplaceholder" in tokenize function<br>

Almost of categories are imbalanced, I used  parameters 'clf__estimator__class_weight': ['balanced', 'balanced_subsample']<br>

Some categories have few message can be remove like child_alone, offer, tool, shop as model don't have data to train.<br>

I want to try more parameters and features, but the training time in my computer take very long time, then I only try with 2 parameters:
1. 'clf__estimator__n_estimators': [20, 40, 60],
2. 'clf__estimator__class_weight': ['balanced', 'balanced_subsample']

Modify the build_model function to lower or remove parameters to get the model train faster

re.sub is not pickle function, then cannot set the n_jobs param

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

disaster_messages and disaster_categories.csv files are provided by Udacity <br>
Otherwise, feel free to use the code here as you would like! <br>
