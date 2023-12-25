import pandas as pd
import numpy as np
import torch
import argparse
from utils import *
from dataset import MLDataset
from models import combined_models, poster_models, ratings_models, title_models 
from metrics import metrics
from torch import optim

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
    parser.add_argument('--checkpoints', default='checkpoints/', help="path to checkpoints folder")

    parser.add_argument('--mode', default='combined', help="combined or single model")
    parser.add_argument('--type', default='', help="model type poster or title or ratings, only using this when --mode==single")
    parser.add_argument('--model', default='CombinedModel', help="model name (make sure to choose a model that match with --mode and --type)")
    
    parser.add_argument('--seed', default=1711, type=int)
    return parser

def main(args):
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 18

    # Load dataframe
    print("Preparing data...")
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
    movies_train = preprocess_df(movies_train)
    movies_test = preprocess_df(movies_test)

    # Add ratings
    movies_train = add_ratings(movies_train, args.data_path + "movie_ratings.txt")
    movies_test = add_ratings(movies_test, args.data_path + "movie_ratings.txt")

    # Preprocess title
    movies_test.loc[:, 'title'] = movies_test['title'].apply(lambda x: preprocess_title(x))
    movies_train.loc[:, 'title'] = movies_train['title'].apply(lambda x: preprocess_title(x))

    # Data loader
    print("Prepare DataLoader...")
    train_set = MLDataset(movies_train, args.data_path)
    test_set = MLDataset(movies_test, args.data_path)

    BATCH_SIZE = args.batch_size
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE)
    print("Finished loading data.")

    print("Start training:")
    training(args, train_dataloader, test_dataloader, num_classes, device)

    print("Start eval:")
    eval(args, test_dataloader, num_classes, device)

def training(args, train_dataloader, test_dataloader, num_classes, device):
    model = getattr(combined_models, args.model)(num_classes)
    mtype = args.type
    if args.mode == "single":
        if mtype == "poster":
            model = getattr(poster_models, args.model)(num_classes)
        elif mtype == "title":
            model = getattr(title_models, args.model)(num_classes)
        elif mtype == "ratings":
            model = getattr(ratings_models, args.model)(num_classes)

    model.to(device)
    lr = args.lr
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    NUM_EP = args.epochs

    checkpoints = args.checkpoints + args.model + ".pt"

    best_f1 = 0
    best_epoch = 0
    best_value_tuple = ()

    for epoch in range(1, NUM_EP+1):
        total_train_loss = 0

        model.train()
        for image, title, ratings, genres in tqdm(train_dataloader):
            image = image.to(device)
            # title = title.to(device)
            ratings = ratings.to(device)
            genres = genres.to(device)

            x = choosing_x(image, ratings, title, args.mode, mtype)

            if (args.mode == "single"):
                out = model(x)
                print("vl single")
            else:
                out = model(*x)
            print(out.shape)
            loss = criterion(out, genres)
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        total_loss_test = 0
        outputs = []
        gts = []

        with torch.no_grad():
            for image, title, ratings, genres in test_dataloader:
                image = image.to(device)
                # title = title.to(device)
                ratings = ratings.to(device)
                genres = genres.to(device)

                x = choosing_x(image, ratings, title, args.mode, mtype)
                loss = criterion(out, genres)
                outputs.append(out)
                gts.append(genres)

                total_loss_test += loss.item()
        
        train_loss = total_train_loss/len(train_dataloader)
        test_loss = total_loss_test/len(test_dataloader)
        outputs = torch.cat(outputs)
        gts = torch.cat(gts)
        f1, acc, recall, precision = metrics(outputs, gts, device)
        print(f'Epoch {epoch}: Train_Loss: {train_loss:^10.3f}|Test Accuracy: {acc:^10.3f}|Precision: {precision:^10.3f}|Recall: {recall:^10.3f}|F1-Score: {f1:^10.3f}')
        
        if f1_all > best_f1:
            best_f1 = f1
            best_epoch = epoch
            best_value_tuple = train_loss, acc, precision, recall
            torch.save(model.state_dict(), checkpoints)
    
    print("Best Values:")
    print(f'Epoch {best_epoch}: Train_Loss: {best_value_tuple[0]:^10.3f}|Test Accuracy: {best_value_tuple[1]:^10.3f}|Precision: {best_value_tuple[2]:^10.3f}|Recall: {best_value_tuple[3]:^10.3f}|F1-Score: {best_f1:^10.3f}')

def eval(args, test_dataloader, num_classes, device):
    model = getattr(models, args.model)(num_classes)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoints + args.model + ".pt"))
    model.eval()

    total_loss_test = 0
    outputs = []
    gts = []
    with torch.no_grad():
        for image, title, ratings, genres in test_dataloader:
            image = image.to(device)
            title = title.to(device)
            ratings = ratings.to(device)
            genres = genres.to(device)

            x = choosing_x(image, ratings, title, args.mode, args.type)
            loss = criterion(out, genres)
            outputs.append(out)
            gts.append(genres)

            total_loss_test += loss.item()

    outputs = torch.cat(outputs)
    gts = torch.cat(gts)
    f1, acc, recall, precision = metrics(outputs, gts, device)    
    test_loss = total_loss_test/len(test_dataloader)
    print(f'Test Accuracy: {acc:^10.3f}|Test Loss: {test_loss:^10.3f}|Precision: {precision:^10.3f}|Recall: {recall:^10.3f}|F1-Score: {f1:^10.3f}')

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)