import copy
import torch
from torch import optim
from torch.autograd import Variable
from torchvision import models
from torch import nn

from . import read_data


class Classifier(object):
    def __init__(self):
        loader, sizes, classes = read_data.get_train_val_loader()
        self.loader = loader
        self.sizes = sizes
        self.classes = classes
        self.model = self.model()

    def get_model(self):
        model_ft = models.resnet18(pretrained=True)
        feature_count = model_ft.fc.in_features
        model_ft.fc = nn.Linear(feature_count, 2)

        if torch.has_cudnn():
            model_ft.cuda()
        return model_ft

    def train(self, scheduler, num_epochs=50):
        criterion = nn.CrossEntropyLoss()

        class_names = self.classes['train']

        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        exp_lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1)

        best_weights = copy.deepcopy(self.model.state_dict())
        best_accuracy = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    self.model.train(True)
                else:
                    self.model.train(False)

                running_loss = 0
                running_corrects = 0

                for data in self.loader[phase]:
                    inputs, labels = [Variable(x) for x in data]

                    optimizer.zero_grad()

                    # forward
                    outputs = self.model(inputs)
                    _, pred = torch.max(outputs, labels)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # stats
                    running_loss += loss.data[0] * inputs.size(0)
                    running_corrects += torch.sum(pred == labels.data)

            epoch_loss = running_loss / self.sizes[phase]
            epoch_acc = running_corrects / self.sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_weights = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_weights)
        return self.model




