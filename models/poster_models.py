import torch
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn


class Resnet(nn.Module):
    def __init__(self, n_classes):
        super(Resnet, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for param in self.resnet.parameters():
            param.requires_grad = False
        in_features = self.resnet.fc.in_features
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-1])
        # print(in_features)
        # self.resnet.fc =  nn.Sequential(
        #                 nn.Linear(in_features, n_classes),
        #                 nn.Sigmoid())
        self.fc1 = nn.Linear(in_features, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(256, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class MultiLabelVGG16(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelVGG16, self).__init__()
        # Load pre-trained VGG-16
        self.vgg16 = models.vgg16(pretrained=True)

        # Modify the last fully connected layer for the number of classes in your problem
        for param in self.vgg16.parameters():
            param.requires_grad = False
        in_features = self.vgg16.classifier[6].in_features
        print(in_features)
        self.vgg16.classifier[6] = nn.Sequential(
                        nn.Linear(in_features, num_classes),
                        nn.Sigmoid())

    def forward(self, x):
        return self.vgg16(x)

class CustomModel(nn.Module):
    def __init__(self, num_classes, input_shape=(3, 256, 256)):
        super(CustomModel, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=(3, 3))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # Layer 2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # Layer 3
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3))
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        # Layer 4
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        # Layer 5
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 14 * 14, 64)
        self.relu5 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        # Layer 6
        self.fc2 = nn.Linear(64, 32)
        self.relu6 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        # Layer 7
        self.fc3 = nn.Linear(32, num_classes)

        self.sigmoid =  nn.Sigmoid()


    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        # Layer 4
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        # Layer 5
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout1(x)

        # Layer 6
        x = self.fc2(x)
        x = self.relu6(x)
        x = self.dropout2(x)

        # Layer 7
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class MultiLabelMobileNetV2(nn.Module):
    def __init__(self, num_labels):
        super(MultiLabelMobileNetV2, self).__init__()
        # Load pre-trained MobileNetV2
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=True)

        # Modify the last fully connected layer for the number of classes in your problem
        for param in self.mobilenet_v2.parameters():
            param.requires_grad = False
        in_features = self.mobilenet_v2.classifier[1].in_features
        self.mobilenet_v2.classifier[1] = nn.Sequential(
                                nn.Linear(in_features, 512),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),

                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),

                                nn.Linear(256, num_labels),
                                nn.Sigmoid()
                              )
    def forward(self, x):
        return self.mobilenet_v2(x)

class MultiLabelDenseNet(nn.Module):
    def __init__(self, num_labels):
        super(MultiLabelDenseNet, self).__init__()
        # Load pre-trained DenseNet
        self.densenet = models.densenet121(pretrained=True)

        # Modify the last fully connected layer for the number of classes in your problem
        for param in self.densenet.parameters():
            param.requires_grad = False
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
                                nn.Linear(in_features, 512),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),

                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),

                                nn.Linear(256, num_labels),
                                nn.Sigmoid()
                              )

    def forward(self, x):
        return self.densenet(x)

class MultiLabelAlexNet(nn.Module):
    def __init__(self, num_labels):
        super(MultiLabelAlexNet, self).__init__()
        # Load pre-trained AlexNet
        self.alexnet = models.alexnet(pretrained=True)

        for param in self.alexnet.parameters():
            param.requires_grad = False
        # Modify the last fully connected layer for the number of classes in your problem
        in_features = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Sequential(
                                nn.Linear(in_features, 512),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),

                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),

                                nn.Linear(256, num_labels),
                                nn.Sigmoid()
                              )

    def forward(self, x):
        return self.alexnet(x)

