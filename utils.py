import os
import json
import pandas as pd
import numpy as np



def check(row):
  return not os.path.exists(row["img_path"])

def preprocess_df(dataframe):
  mask = dataframe.apply(check, axis=1)

  data = dataframe.drop(dataframe[mask].index)
  return data


""" 
The code for convert from user's rating to 
movie's rating take a lot of time to run, 
so i only run it once and save the output 
in a .txt file that can be download from 
this link: https://drive.google.com/uc?id=1rvmwYsRU9IykFPTlsckDNm4RHO1CxcJ3
make sure to put it in the data/ folder
"""
def add_ratings(df, path):
  with open(path, 'r') as file:
    loaded_dict = json.load(file)
  loaded_dict_converted = {int(k): np.array(v) if isinstance(v, list) else v for k, v in loaded_dict.items()}
  df['ratings'] = df['movieid'].map(loaded_dict_converted)
  return df

def preprocess_title(text):
    text = text.lower()
    text = text[:-5].strip()  # remove (year)
    text = text[:text.find('(')].strip() 
    # rearrange the title
    if len(text.split(',')) > 1:
        text = text.split(',')[1].strip() + ' ' + text.split(',')[0].strip()
    return text

def choosing_x(image, ratings, title, mode, mtype):
  if mode == "single":
    if mtype == "poster":
        return image
    elif mtype == "title":
        return title
    elif mtype == "ratings":
        return ratings