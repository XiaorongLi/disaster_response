import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    '''Function loads messages and categories, then merges on id'''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    data = pd.merge(messages, categories, on='id')
    return data


def clean_data(df):
    '''Function cleans and converts the categories into an encoding matrix ; 
    drops duplicates.
    
    Parameters
    ----------
    df : mereged dataframe returned by load_data()
    '''
    #create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    
    #rename the new columns
    row = categories.iloc[0,:]
    category_colnames = [x[0] for x in row.str.split('-')]
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        
        # set each value to be the last character of the string
        categories[column] = [x[1] for x in categories[column].str.split('-')]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    #Notice some values in the 'related' column are 2, 
    #which is assumed to be an error. Replace them by 1.
    categories.iloc[:,0].replace(2,1, inplace=True)
    
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    '''Function saves the cleaned data into a sqlite database

    Parameters
    ----------
    df : cleaned database
    database_filename: string, database filepath

    Returns
    -------
    None.

    '''
    string = 'sqlite:///' + database_filename
    engine = create_engine(string)
    
    table_name = string.split('.')[0]
    #sql = 'DROP TABLE IF EXISTS ' + table_name + ';'
    #engine.execute(sql)
    df.to_sql(table_name, engine, index=False, )
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()