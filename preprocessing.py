import os
import pandas as pd
import numpy as np

def process_articles(articles, articles_dict_keys):
    """
    Process a list of articles to extract relevant information based on predefined keys.

    This function splits each article into lines, extracts information based on the specified keys,
    and accumulates the information in a dictionary format suitable for DataFrame conversion.

    Parameters:
        articles (list): A list of articles in string format.
        articles_dict_keys (list): A list of keys to extract information from the articles.

    Returns:
        dict: A dictionary with keys as specified by articles_dict_keys and values as lists of extracted information.
    """
    articles_dict = {key: [] for key in articles_dict_keys}
    for article in articles:
        lines = article.split("\n")
        article_data = {key: [] for key in articles_dict_keys}
        current_key = None
        for line in lines:
            if len(line) > 4 and line[4] == '-':
                key, value = line.split("-", 1)
                key, value = key.strip(), value.strip()
                if key in article_data:
                    article_data[key].append(value)
                    current_key = key
            elif current_key:
                article_data[current_key][-1] += ' ' + line.strip()
        for key in articles_dict_keys:
            articles_dict[key].append(' '.join(article_data[key]) if article_data[key] else None)
    return articles_dict

def create_dataframe(articles_dict, columns):
    """
    Create a pandas DataFrame from a dictionary of articles.

    Parameters:
        articles_dict (dict): A dictionary where keys correspond to DataFrame columns and values are lists of column values.
        columns (list): A list of column names for the DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame constructed from the articles_dict with specified columns.
    """
    return pd.DataFrame(articles_dict, columns=columns)

def clean_dataframe(df):
    """
    Clean the DataFrame by removing rows with missing values and empty strings in the 'AB' column and duplicate PMIDs.

    Additionally, resets the DataFrame index after dropping rows.

    Parameters:
        df (pandas.DataFrame): The DataFrame to clean.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    df.replace('', np.nan, inplace=True)

    print(f"Number of missing values in AB column: {df['AB'].isnull().sum()}")
    df = df.dropna(subset=['AB'])
    print(f"Number of articles after dropping missing values in AB column: {df.shape[0]}")

    print(f"Number of duplicate PMIDs: {df['PMID'].duplicated().sum()}")
    df = df.drop_duplicates(subset=['PMID'])
    print(f"Number of articles after dropping duplicates: {df.shape[0]}")
    
    df = df.reset_index(drop=True)
    return df

def convert_data_types(df):
    """
    Convert data types of columns in a DataFrame.

    Converts specified columns to string type and specified columns to datetime type,
    with error handling for datetime conversion.

    Parameters:
        df (pandas.DataFrame): The DataFrame whose column data types are to be converted.

    Returns:
        pandas.DataFrame: The DataFrame with converted column data types.
    """
    # Columns to be converted to string
    string_columns = ['PMID', 'TI', 'AB', 'FAU', 'AU', 'MH', 'OT', 'PL', 'JT', 'PMC', 'DP', 'EDAT', 'MHDA']
    for column in string_columns:
        df[column] = df[column].astype(str)

    return df

def select_important_columns(df, important_keys):
    """
    Select and return important columns from the DataFrame based on a list of keys.

    Parameters:
        df (pandas.DataFrame): The source DataFrame from which to select columns.
        important_keys (list): A list of column names to keep in the resulting DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing only the columns specified in important_keys.
    """
    df_important = df[important_keys]
    return df_important

def save_to_parquet(df, project_dir, filename='pubmed_intelligence_abstracts_cleaned.parquet'):
    """
    Save a DataFrame to a Parquet file in the specified directory with the given filename.

    Parameters:
        df (pandas.DataFrame): The DataFrame to save.
        project_dir (str): The directory to save the Parquet file in.
        filename (str): The name of the Parquet file. Defaults to 'pubmed_intelligence_abstracts_cleaned.parquet'.

    Prints:
        A confirmation message with the path to the saved Parquet file.
    """
    file_path = os.path.join(project_dir, 'data', 'cleanData', filename)
    df.to_parquet(file_path, index=False)
    print(f"DataFrame saved to {file_path}")

def load_from_parquet(file_path):
    """
    Load a DataFrame from a Parquet file.

    Parameters:
        file_path (str): The path to the Parquet file to load.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    df = pd.read_parquet(file_path)
    return df