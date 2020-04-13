import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    params:
    - messages_filepath: path to CSV file that contains the messages
    - categories_filepath: path to a CSV file that contains the categories

    returns:
    - df: a dataframe that contains the messages and categories combined.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #Merge datasets.
    df = pd.merge(messages, categories, on="id")
    
    return df





def clean_data(df):
    """
    params:
    - df: a dataframe containing the messages and their categories 

    returns:
    - a dataframe with cleaned data; categories are each represented by a column; duplicates dropped.
    """

    #Split `categories` into separate category columns.
    categories = df.categories.str.split(";", expand=True)
    new_col_names = {col : categories[col][0].split("-")[0] for col in categories.columns}
    categories.rename(columns=new_col_names, inplace=True)

    #Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)


    #Replace `categories` column in `df` with new category columns.

    # drop the original categories column from `df`
    df.drop(columns=["categories"], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    #Remove duplicates and return
    return df.drop_duplicates()





def save_data(df, database_filename):
    """
    params:
    - df: dataframe to be saved into a database
    - database_filename: name of the database file in which the data will be stored

    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False)




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