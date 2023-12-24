import torch
from torchmetrics.classification import MultilabelF1Score, MultilabelRecall, MultilabelPrecision, MultilabelAccuracy

num_classes = 18

def metrics(preds, gts):
    f1 = MultilabelF1Score(num_labels=num_classes, average='macro', threshold=0.8)
    f1 = f1.to(device)
    recall = MultilabelRecall(num_labels=num_classes, average='macro', threshold=0.8)
    recall = recall.to(device)
    precision = MultilabelPrecision(num_labels=num_classes, average='macro', threshold=0.8)
    precision = precision.to(device)
    accuracy = MultilabelAccuracy(num_labels=num_classes, average='macro', threshold=0.8)
    accuracy = accuracy.to(device)

    f1_score = f1(preds, gts)
    acc = accuracy(preds, gts)
    recall_ = recall(preds, gts)
    precision_ = precision(preds, gts)

    return f1_score, acc, recall_, precision_