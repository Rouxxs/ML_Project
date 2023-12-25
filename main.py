import pandas as pd
import numpy as np
import torch
import argparse
from utils import *
from dataset import MLDataset

from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=2e-4, type=float, help="learning rate setting")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--data_path', default='data/', help="path to the dataset folder")
    parser.add_argument('--output', default='output/', help="path to output folder")
    parser.add_argument('--output', default='checkpoints/', help="path to checkpoints folder")


    parser.add_argument('--mode', default='combined', help="combined or single model")
    parser.add_argument('--type', default='', help="model type, only using this when --mode==single")
    parser.add_argument('--model', default='CombinedModel', help="model name (make sure to choose a model that match with --mode and --type)")
    
    parser.add_argument('--seed', default=1711, type=int)
    return parser

def main(args):
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 18

    # Load dataframe
    movies_train = pd.read_csv(args.data_path + 'ml1m/content/dataset/movies_train.dat', engine='python',
                         sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
    movies_test = pd.read_csv(args.data_path + 'ml1m/content/dataset/movies_test.dat', engine='python',
                         sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')

    movies_train['genre'] = movies_train.genre.str.split('|')
    movies_test['genre'] = movies_test.genre.str.split('|')

    # Add img path
    folder_img_path = args.data_path + 'ml1m/content/dataset/ml1m-images'
    movies_train['id'] = movies_train.index
    movies_train.reset_index(inplace=True)
    movies_train['img_path'] = movies_train.apply(lambda row: os.path.join(folder_img_path, f'{row.id}.jpg'), axis = 1)

    movies_test['id'] = movies_test.index
    movies_test.reset_index(inplace=True)
    movies_test['img_path'] = movies_test.apply(lambda row: os.path.join(folder_img_path, f'{row.id}.jpg'), axis = 1)

    # Remove rows that the corresponding img does not exit
    movies_train = get_dataframe(movies_train)
    movies_test = get_dataframe(movies_test)

    # Add ratings
    movies_train = add_ratings(movies_train, args.data_path)
    movies_test = add_ratings(movies_test, args.data_path)

    # Preprocess title
    movies_test.loc[:, 'title'] = movies_test['title'].apply(lambda x: preprocess_title(x))
    movies_train.loc[:, 'title'] = movies_train['title'].apply(lambda x: preprocess_title(x))

    # Data loader
    train_set = MLDataset(movies_train)
    test_set = MLDataset(movies_test)

    BATCH_SIZE = args.batch_size
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE)

