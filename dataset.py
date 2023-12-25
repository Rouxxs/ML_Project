import os
import torch
import json
import cv2
import pandas as pd


from torch.utils.data.dataset import Dataset


class MLDataset(Dataset):
    def __init__(self, data, data_path):
        self.data = data
        # label genre
        with open(data_path + 'ml1m/content/dataset/genres.txt', 'r') as f:
            genre_all = f.readlines()
            genre_all = [x.replace('\n','') for x in genre_all]
        self.genre2idx = {genre:idx for idx, genre in enumerate(genre_all)}

    def __getitem__(self, index):
        img_path = self.data.iloc[index].img_path
        genre = self.data.iloc[index].genre
        ratings = self.data.iloc[index].ratings
        title = self.data.iloc[index].title

        # preprocess img
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
        else:
            img = np.random.rand(256,256,3)
        img = cv2.resize(img, (256,256))
        img_tensor = torch.from_numpy(img.transpose(2,0,1)).float()

        # preprocess label
        genre_vector = np.zeros(len(self.genre2idx))

        for g in genre:
            genre_vector[self.genre2idx[g]] = 1
        genre_tensor = torch.from_numpy(genre_vector).float()

        ratings_tensor = torch.from_numpy(ratings).float()

        return img_tensor, title, ratings_tensor, genre_tensor

    def __len__(self):
        return len(self.data)