import numpy as np
import pandas as pd
from ast import literal_eval
import os

def remove_rows_with_missing_ratings(raw_data):
    raw_data.dropna(subset=["Cleanliness_rating","Accuracy_rating","Communication_rating","Location_rating","Check-in_rating","Value_rating"], inplace=True)
    raw_data.drop(raw_data.columns[-1], axis = 1, inplace=True)
    return raw_data

def combine_description_strings(dc):
    dc_list = literal_eval(dc.strip())[1:]
    dc_list_without_blank_quotes = list(filter(lambda x: x != "", dc_list))
    full_dc = ' '.join(dc_list_without_blank_quotes)
    return full_dc

def accepted_description(description):
    try: 
        description_type = type(literal_eval(description))
        if description_type == list:
            return description
        else:
            return np.nan
    except:
        return np.nan

def fix_description_strings(df):
    df['Description'] = df['Description'].apply(accepted_description)
    df.dropna(subset=["Description"], inplace=True)
    df["Description"] = df["Description"].apply(combine_description_strings)
    return df

def set_default_feature_values(df): 
    for column_name in ['guests','beds','bathrooms','bedrooms']: 
        df.loc[df[column_name].isna(), column_name] = 1 
    return df

def clean_tabular_data(df):
    raw_data_with_ratings = remove_rows_with_missing_ratings(df)
    raw_data_with_description = fix_description_strings(raw_data_with_ratings)
    raw_data_default_features = set_default_feature_values(raw_data_with_description)
    return raw_data_default_features

def load_airbnb(df,label): 
    features = df[label][df[label] != label]
    return (features,label)

if __name__ == '__main__':
    raw_data = pd.read_csv("/Users/vikasiniperemakumar/Desktop/AiCore/airbnb-property-listings/tabular_data/listing.csv", index_col=0)
    clean_data = clean_tabular_data(raw_data)
    os.chdir("/Users/vikasiniperemakumar/Desktop/AiCore/airbnb-property-listings/tabular_data")
    clean_data.to_csv("clean_tabular_data.csv")
   
