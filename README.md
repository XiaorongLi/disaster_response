# disaster_response

## Background
During severe disasters, the disaster response oganizations aim at providing necessary salvation and support in time for targets as precisely as possible. Data from various social media serve as an important channel for the organizations to acquire relevant information which can support with decision-making. For example, a community could suffer from food and drinkable water shortage during a hurricane and some people may post on their social media softwares, searching for help: 'We have been trapped here during the hurricane and don't have enough food!' And their can be numerous messages posted on the social media. The response oganizations want to automatically detect the relevant messages and figure out if someone is in trouble and what kind of help they need, through a proper machine-learning model. This project developed just such a model, expecting to help the organizations improve their performance.

## Hightlights
- Development of a complete ETL-NLP-ML pipeline with Python Scikit-learn for 46,000+ messages collected from social media
- Multilable classification of data using random forest and random forest in XGboost
- Parameter search with cross validation

## Files
In `data` folder:
- `disaster_messages.csv` Messages collected from social media
- `disaster_categories.csv` Category information of the messages
- `message.db` Cleaned data
- `process_data.py` Python script for data processing (refactored from Jupyter notebooks, see below)

In `models` folder:
- `train_classifier.py` Python script for NLP-ML models (refactored from Jupyter notebooks, see below)

In `JupyterNotebook` folder:
- `ETL_Pipeline_Preparation.ipynb` Processing data
- `ML_Pipeline_Preparation.ipynb` Building NLP-ML pileline

In `app` folder:
- files intended to build a web app, have not finished yet...
