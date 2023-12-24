import torch
import torchvision.models as models
import torch.nn as nn

class BertFeatureExtractor(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(BertFeatureExtractor, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def forward(self, input_text):
        # Tokenize and encode the input text
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)

        # Move tensors to the desired device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state
        return embeddings

class BertMultiLabelClassifier(nn.Module):
    def __init__(self, num_labels, bert_feature_dim=768):
        super(BertMultiLabelClassifier, self).__init__()
        self.bert_extractor = BertFeatureExtractor()
        self.classifier = nn.Sequential(
              nn.Linear(bert_feature_dim, 512),
              nn.ReLU(),
              nn.Dropout(p=0.2),

              nn.Linear(512, 256),
              nn.ReLU(),
              nn.Dropout(p=0.2),

              nn.Linear(256, num_labels),
              nn.Sigmoid()
        )

    def forward(self, input_text):
        bert_features = self.bert_extractor(input_text)
        cls_representation = bert_features[:, 0, :]
        x = self.classifier(cls_representation)
        return x