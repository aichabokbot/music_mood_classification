from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

from PIL import Image
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    """
    Function to train our Pytorch model
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def map_from_idx(array, map_list):
    """
    Function that takes an array of indices positions and returns 
    the elements at these specific positions in map_list
    """
    res = []
    for i in array:
        res.append(map_list[int(i)])
    return res


def predict(model, img_path, loader, class_names, k=2):
    """
    Function to compute the output of the Pytorch model for one image
    Returns the probability and the predicted class
    """
    img = Image.open(img_path)
    img = loader(img).unsqueeze(0).to(device)
    model.eval()
    output = model(img)
    proba, class_idx = torch.topk(output, k)
    proba = proba[0].detach().numpy()
    class_idx = class_idx[0].detach().numpy()
    class_val = map_from_idx(class_idx, class_names)
    return proba, class_val