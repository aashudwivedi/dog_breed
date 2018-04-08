import copy
import torch
from torch import optim
from torch.autograd import Variable
from torchvision import models
from torch import nn

import read_data


class Classifier(object):
    def __init__(self):
        loader, sizes, classes = read_data.get_train_val_loader()
        self.loader = loader
        self.sizes = sizes
        self.model = self.get_model(classes)

    def get_model(self, classes):
        model_ft = models.resnet18(pretrained=True)
        feature_count = model_ft.fc.in_features
        model_ft.fc = nn.Linear(feature_count, classes)

        if torch.has_cudnn:
            model_ft.cuda()
        return model_ft

    def train(self, num_epochs=50):
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        scheduler = optim.lr_scheduler.StepLR(
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
                    inputs = Variable(data[0])
                    labels = Variable(data[1])

                    if torch.has_cudnn:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    optimizer.zero_grad()

                    # forward
                    # import ipdb; ipdb.set_trace()
                    outputs = self.model(inputs)
                    _, pred = torch.max(outputs.data, 1)

                    loss = criterion(outputs, labels.view(labels.shape[0]))

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

            print('{"metric": "loss", "{}": <float>}'.format(epoch_loss))
            print('{"metric": "acc", "{}": <float>}'.format(epoch_acc))

            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_weights = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_weights)
        return self.model


if __name__ == '__main__':
    classifier = Classifier()
    classifier.train()




