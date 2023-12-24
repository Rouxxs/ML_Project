import os
import json
import pandas as pd


def check(row):
  return not os.path.exists(row["img_path"])

def preprocess_df(dataframe):
  mask = dataframe.apply(check, axis=1)

  data = dataframe.drop(dataframe[mask].index)
  return data

def add_ratings(df, path):
  with open(path, 'r') as file:
    loaded_dict = json.load(file)
  loaded_dict_converted = {int(k): np.array(v) if isinstance(v, list) else v for k, v in loaded_dict.items()}
  df['ratings'] = df['movieid'].map(loaded_dict_converted)
  return df

def preprocess_title(text):
    text = text.lower()
    text = text[:-5].strip()
    text = text[:text.find('(')].strip()
    if len(text.split(',')) > 1:
        text = text.split(',')[1].strip() + ' ' + text.split(',')[0].strip()
    return text