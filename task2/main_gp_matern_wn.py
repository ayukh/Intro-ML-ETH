# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, WhiteKernel

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    #print("Training data:")
    #print("Shape:", train_df.shape)
    #print(train_df.head(2))
    #print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    #print("Test data:")
    #print(test_df.shape)
    #print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train   
    X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df)

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    
    # Impute missing data using KNN Imputer for train set

    train_new = train.dropna(subset=['price_CHF'])

    num_cols = train_new.columns[1:]

    imputer = KNNImputer(n_neighbors=5)
    train_new[num_cols] = imputer.fit_transform(train_new[num_cols])
    
    # Either drop nan rows or impute for test
    test_new = test.copy()
    num_cols = test_new.columns[1:]

    imputer = KNNImputer(n_neighbors=5)
    test_new[num_cols] = imputer.fit_transform(test_new[num_cols])
    
    # Encode categorical variable (season)
    train_new["season"] = train_new["season"].astype('category')
    train_new["season"] = train_new["season"].cat.codes

    test_new["season"] = test_new["season"].astype('category')
    test_new["season"] = test_new["season"].cat.codes

    ind_cols = ['season', 'price_AUS', 'price_CZE', 'price_GER',
           'price_ESP', 'price_FRA', 'price_UK', 'price_ITA', 'price_POL', 'price_SVK']
    dep_col = 'price_CHF'
        
    # Import new data as train/test for modelling
    X_train = train_new[ind_cols]
    y_train = train_new[dep_col]
    X_test = test_new

    print("X train shape: ", X_train.shape)
    print("Y train shape: ", Y_train.shape)
    print("X test shape: ", X_test.shape)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions

    kernel = Matern(nu=0.5, length_scale=1.0) + WhiteKernel()

    gp_model = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', random_state=None, n_restarts_optimizer=0, normalize_y=False)
    gp_model.fit(X_train, y_train)

    y_pred = gp_model.predict(X_test)
    
    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")