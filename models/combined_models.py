import torch
import torchvision.models as models
import torch.nn as nn
from title_models import BertFeatureExtractor
from ratings_model import RatingsExtractModel

# Concat
class CombinedModel(nn.Module):
  def __init__(self, num_labels, bert_feature_dim=768):
    super(CombinedModel, self).__init__()
    rating_features = 256
    # Image
    self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in self.resnet.parameters():
      param.requires_grad = False
    resnet_infeatures = self.resnet.fc.in_features
    self.image_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

    # Ratings
    self.ratings_extractor = RatingsExtractModel()

    # Title
    self.bert_extractor = BertFeatureExtractor()

    # Classifier
    self.classifier = nn.Sequential(
      nn.Linear(bert_feature_dim + resnet_infeatures + rating_features, 512),
      nn.ReLU(),
      nn.Dropout(p=0.2),

      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Dropout(p=0.2),

      nn.Linear(256, num_labels),
      nn.Sigmoid()
    )

  def forward(self, image, rating, title):
    # title
    title_features = self.bert_extractor(title)
    title_features = title_features[:, 0, :]
    # print(title_features.shape)
    # image
    image_features = self.image_extractor(image)
    image_features = image_features.reshape(image_features.shape[:2])
    # print(image_features.shape)
    # ratings
    ratings_features = self.ratings_extractor(rating)
    # print(ratings_features.shape)
    combined_features = torch.cat([title_features, image_features, ratings_features], dim=1)

    output = self.classifier(combined_features)
    return output


# Weight
class CombinedModelWeight(nn.Module):
  def __init__(self, num_labels, bert_feature_dim=768, weights=[0.05, 0.9, 0.05]):
    super(CombinedModelWeight, self).__init__()
    rating_features = 256
    # Image
    self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in self.resnet.parameters():
      param.requires_grad = False
    resnet_infeatures = self.resnet.fc.in_features
    self.image_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

    # Ratings
    self.ratings_extractor = RatingsExtractModel()

    # Title
    self.bert_extractor = BertFeatureExtractor()

    # Linear transformations to ensure consistent output sizes
    self.linear1 = nn.Linear(resnet_infeatures, 256)
    # self.linear2 = nn.Linear(rating_features, 256)
    self.linear3 = nn.Linear(bert_feature_dim, 256)

    # Weight
    self.weights = torch.tensor(weights, dtype=torch.float32)
    self.weights /= self.weights.sum()

    # Classifier
    self.classifier = nn.Sequential(
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(p=0.2),

      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Dropout(p=0.2),

      nn.Linear(64, num_labels),
      nn.Sigmoid()
    )

  def forward(self, image, rating, title):
    # title
    title_features = self.bert_extractor(title)
    title_features = title_features[:, 0, :]
    feat3 = self.linear3(title_features)
    # print(title_features.shape)
    # image
    image_features = self.image_extractor(image)
    # print(image_features.shape)
    image_features = image_features.reshape(image_features.shape[:2])
    feat1 = self.linear1(image_features)
    # ratings
    ratings_features = self.ratings_extractor(rating)
    # feat2 = self.linear2(ratings_features)
    # print(ratings_features.shape)
    combined_features = (
        self.weights[0] * feat1 +
        self.weights[1] * ratings_features +
        self.weights[2] * feat3
    )

    output = self.classifier(combined_features)
    return output