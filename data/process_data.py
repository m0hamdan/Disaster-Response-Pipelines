import sys
import pandas as pd 
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges the provided messages and categories datasets
    
    Args:
        messages_filepath (str): path to the messages dataset
        categories_filepath (str): path to the categories dataset
        
    Returns:
        A new dataset 
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id')
    


def clean_data(df):
    """
    Cleans the merged data frame
    
    Args:
        df (Dataframe): path to the messages dataset
        
    Returns:
        A cleaned data frame 
    """
    #extract a list of new column names for categories
    categories = df['categories'].str.split(';',expand=True)
    first_row = categories.iloc[0]
    category_colnames = first_row.apply(lambda x: x.split('-')[0] )
    categories.columns = category_colnames
    categories.related[categories.related == 'related-2'] = 'related-1'
    #Iterate through the category columns in df to keep only the last character of each string (the 1 or 0)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  categories[column].apply(lambda x: x.split('-')[1] )
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df=df.drop('categories', axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df =  pd.concat([df, categories], axis=1, sort=False)
    # drop duplicates
    df = df.drop_duplicates(keep='first')
    return df
    


def save_data(df, database_filename):
    """
    Saves the provided data frame into the specified database
    
    Args:
        df (Dataframe): the messages dataframe
        database_filename (str): database path
        
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Messages', engine, index=False,if_exists='replace')
      


def main():
    """
    Main function
    
    this function will execute the ETL pipeline:
    1- invoke the 'load_data' function above and
    2- pass the data to the 'clean_data', then
    3- save the cleaned data frame into sqlite database by calling 'save_data' function 
        
    """
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