# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 111
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def generate_embeddings(batch_size=512):
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    
    # TODO: define a transform to pre-process the images
    train_transforms = transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root="/kaggle/input/imltask3/dataset/dataset/", transform=train_transforms)
    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't 
    # run out of memory
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, # tune this
                              shuffle=False,
                              pin_memory=True, num_workers=2)
    # TODO: define a model for extraction of the embeddings (Hint: load a pretrained model,
    #  more info here: https://pytorch.org/vision/stable/models.html)
    model = resnet50(weights="IMAGENET1K_V2")

    embeddings = []
    embedding_size = list(model.children())[-1].in_features # 2048
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))

    # TODO: Use the model to extract the embeddings. Hint: remove the last layers of the 
    # model to access the embeddings the model generates.
     
    model.eval() 

    model = nn.Sequential(*list(model.children())[:-1]) # remove last layer of the model

    print('Extracting features:')
    with torch.no_grad(): 
        for batch_idx, (image, image_idx) in enumerate(tqdm(train_loader)):
            embed_features = model(image) # get features from pretrained model  
            embed_features = embed_features.squeeze().cpu().numpy() # get to shape (256, 2048)
            embeddings[batch_idx * train_loader.batch_size : (batch_idx + 1) * train_loader.batch_size] = embed_features           
            
    np.save(f'embeddings_{batch_size}.npy', embeddings)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/",
                                         transform=None)



    filenames = np.loadtxt("/kaggle/input/imltask3/filenames.txt", dtype=str) # if run on kaggle
    embeddings = np.load('/kaggle/input/imltask3/embeddings_512.npy') # if run on kaggle
    # TODO: Normalize the embeddings across the dataset
    
    embeddings = (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)
    
    file_to_embedding = {}
    for i in range(len(filenames)):
        file_name = filenames[i]
        file_to_embedding[file_name] = embeddings[i]
        
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in tqdm(triplets):
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 2):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader

# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self, dropout=True, dropout_p=0.5): #0.4 -> tune this
        """
        The constructor of the model.
        """
        super().__init__()
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1)
        )
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1)
        )
        
        self.fullycon1 = nn.Sequential(nn.Linear(8 * 2 * 1, 120), nn.ReLU())
        self.fullycon2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        if dropout:
            self.fullycon3 = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(84, 64))
        else:
            self.fullycon3 = nn.Linear(84, 64)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = x.view(-1, 8 * 2 * 1)
        x = self.fullycon1(x)
        x = self.fullycon2(x)
        x = self.fullycon3(x)
        return x

def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)
    print('device: ', device)
    n_epochs = 1
    batch_size = 256

    # TODO: define a loss function, optimizer and proceed with training. Hint: use the part
    # of the training data as a validation split. After each epoch, compute the loss on the
    # validation
    # split and print it out. This enables you to see how your model is performing
    # on the validation data before submitting the results on the server. After choosing the
    # best model, train it on the whole training data.

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    print("Train model")

    model.train()
    for epoch in range(n_epochs):
        train_loss_epoch = []
        train_acc_epoch = []
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                loss = criterion(output, target)
                train_loss_epoch.append(loss.item())
                correct = (predictions == target).sum().item()
                accuracy = correct / len(predictions)
                train_acc_epoch.append(accuracy)

                loss.backward()
                optimizer.step()

                train_loss_avg = np.sum(train_loss_epoch) / len(train_loss_epoch)
                train_acc_avg = np.sum(train_acc_epoch) / len(train_acc_epoch)
                tepoch.set_postfix({'Train loss': train_loss_avg, 'Train accuracy': 100. * train_acc_avg})

    return model


def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch = x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.argmax(dim=1, keepdim=True).squeeze().cpu().numpy()
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.hstack(predictions)
    np.savetxt("results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = '/kaggle/input/imltask3/train_triplets.txt'
    TEST_TRIPLETS = '/kaggle/input/imltask3/test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('/kaggle/input/imltask3/embeddings_512.npy') == False):
        generate_embeddings(batch_size=512)

    # load the training and testing data
    X, y = get_data(TRAIN_TRIPLETS)
    X = X.reshape(-1, 512, 4, 3)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)
    X_test = X_test.reshape(-1, 512, 4, 3) # resize to 4dTensor for CNNs

    # Create data loaders for the training and testing data
    train_loader = create_loader_from_np(X, y, train = True, batch_size=64)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False)

    # define a model and train it
    model = train_model(train_loader)
    
    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")
