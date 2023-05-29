# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
from pprint import pprint
from sklearn.model_selection import LeaveOneOut
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import scale
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression, Lasso

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None

    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("public/pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles",
                                                                                                         axis=1).to_numpy()
    y_pretrain = pd.read_csv("public/pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("public/train_features.csv.zip", index_col="Id", compression='zip').drop("smiles",
                                                                                                   axis=1).to_numpy()
    y_train = pd.read_csv("public/train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("public/test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """

    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraining data
        # and then used to extract features from the training and test data.

        # activation = nn.ReLU()

        # THINGS TO INCLUDE: BATCH NORM, DROPOUT
        self.encoder = nn.Sequential(
            nn.Linear(1000, 500), nn.BatchNorm1d(500),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(500, 250), nn.BatchNorm1d(250),
            nn.Dropout(0.3),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(250, 90), nn.BatchNorm1d(90),
            # nn.Sigmoid()
            # nn.ReLU(),
            # nn.Linear(100, 50), nn.BatchNorm1d(50),
            # nn.ReLU(),
            # nn.Linear(50, 10), nn.BatchNorm1d(50),
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # nn.Linear(10, 50), nn.BatchNorm1d(50), nn.ReLU(),
            # nn.Linear(50, 100), nn.BatchNorm1d(100), nn.ReLU(),
            nn.Linear(90, 250), nn.BatchNorm1d(250),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(250, 500), nn.BatchNorm1d(500),
            nn.Dropout(0.3),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(500, 1000)
        )
        self.last_linear = nn.Sequential(nn.Linear(90, 1))

    def forward(self, x):
        """
        The forward pass of the model.
        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture
        # defined in the constructor.
        x = self.encoder(x)
        y = self.last_linear(x)
        x = self.decoder(x)
        return x, y
    
    
def make_feature_extractor(x, y, batch_size=256, eval_size=1000):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set

    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    # model declaration
    model = Net()
    model.train()

    # TODO: Implement the training loop. The model should be trained on the pretraining data. Use validation set
    # to monitor the loss.

    criterion_decoded = nn.MSELoss()
    criterion_predictions = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    n_epochs = 5
    a = 0.15

    losses = []
    valid_losses = []

    train_loader = DataLoader(
        dataset=TensorDataset(x_tr, y_tr),
        batch_size=batch_size,
        shuffle=True
    )

    valid_loader = DataLoader(
        dataset=TensorDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=True
    )

    for epoch in range(n_epochs):
        train_loss_epoch = []
        valid_loss_epoch = []
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch} train")

                optimizer.zero_grad()

                decoded_features, predictions = model(data)
                predictions = predictions.squeeze()

                loss_decoded = criterion_decoded(decoded_features, data)
                loss_predictions = criterion_predictions(predictions, target)

                train_loss = a*loss_predictions + (1-a)*loss_decoded

                train_loss_epoch.append(train_loss.item())

                train_loss.backward()
                optimizer.step()

                tepoch.set_postfix({'Train loss': train_loss.item()})

        train_loss_avg = np.mean(train_loss_epoch)

        with torch.no_grad():
            with tqdm(valid_loader, unit="batch") as tepoch:
                for valid_data, valid_target in tepoch:
                    tepoch.set_description(f"Epoch {epoch} valid")

                    valid_decoded_features, valid_predictions = model(valid_data)
                    valid_predictions = valid_predictions.squeeze()

                    valid_loss_decoded = criterion_decoded(valid_decoded_features, valid_data)
                    valid_loss_predictions = criterion_predictions(valid_predictions, valid_target)

                    valid_loss = a*valid_loss_predictions + (1-a)*valid_loss_decoded

                    valid_loss_epoch.append(valid_loss.item())

                    tepoch.set_postfix({'Validation loss': valid_loss.item()})

        valid_loss_avg = np.mean(valid_loss_epoch)


        losses.append(train_loss_avg)
        valid_losses.append(valid_loss_avg)

        print('Final train loss: ', train_loss_avg, 'Final valid loss: ', valid_loss_avg)

    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        model.eval()
        # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
        model_no_last_layers = nn.Sequential(*list(model.children())[:-2])

        with torch.no_grad():
            if isinstance(x, pd.DataFrame):
                x = x.to_numpy()
            x = torch.tensor(x, dtype=torch.float)
            x_features = model_no_last_layers(x)

        return x_features

    return make_features


if __name__ == '__main__':
    
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    
    # Utilize pretraining data by creating feature extractor which extracts lumo energy
    # features from available initial features
    feature_extractor = make_feature_extractor(x_pretrain, y_pretrain)

    x_train_transformed = feature_extractor(x_train).numpy()
    x_test_transformed = feature_extractor(x_test).numpy()

    y_pred = np.zeros(x_test.shape[0])
    
    regression_model = Ridge(alpha=0.5)
    
    regression_model.fit(x_train_transformed, y_train)
    y_pred = regression_model.predict(x_test_transformed)

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")