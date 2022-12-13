import numpy as np
import pandas as pd


def load_data(data_path):
    """
    This function loads the data in data_path csv into two numpy arrays:
    features (of size NxK) and targets (of size Nx1) where N is the number of rows
    and K is the number of features.

    Args:
        data_path (str): path to csv file containing the data

    Output:
        features (np.array): numpy array of size NxK containing the K features
        targets (np.array): numpy array of size 1xN containing the N targets.
        attribute_names (list): list of strings containing names of each attribute
            (headers of csv)
    """
    if data_path.endswith('gz'):
        df = pd.read_csv(data_path, compression='gzip')
    else:
        df = pd.read_csv(data_path)

    feature_columns = [col for col in df.columns if col != "class"]
    features = df[feature_columns].to_numpy()
    target = df[["class"]].to_numpy()

    return features, target, feature_columns


def train_test_split(features, targets, fraction):
    """
    Split features and targets into training and testing. The first M points
    from the data will be used for training and the remaining
    (features.shape[0] - M) points will be used for testing. Where M is:

        M = int(features.shape[0] * fraction)

    However, when fraction is 1.0, both training and test splits are
    the entire dataset. Code for this special case is provided for you.

    Args:
        features (np.array): NxD numpy array containing D features for each example
        targets (np.array): Nx1 numpy array containing labels corresponding to each example
        fraction (float between 0.0 and 1.0): fraction of examples to be drawn for training

    Returns (a tuple containing four variables):
        train_features: MxD numpy array of examples to be used for training
        train_targets: Mx1 numpy array of targets corresponding to `train_features`
        test_features: (N - M)xD numpy array of examples to be used for testing
        test_targets: (N - M)x1 numpy array of targets corresponding to `test_features`
    """

    if (fraction > 1.0):
        raise ValueError('N cannot be bigger than number of examples!')
    elif fraction == 1.0:
        return features, targets, features, targets
    
    # Get shape of data matrix
    N = features.shape[0]
    D = features.shape[1]
    M = int(N*fraction)

    # print("features.shape: ", features.shape)
    # print("fraction: ", fraction)
    # print("M: ", M)
    # print("D: ", D)

    train_features = features[:M,:D]
    test_features = features[M:N,:D]
    train_targets = targets[:M]
    test_targets = targets[M:N]

    # print("features.shape: ", features.shape)
    # print("targets.shape: ", features.shape)
    # print("train_features.shape: ", train_features.shape)
    # print("test_features.shape: ", test_features.shape)
    # print("train_targets.shape: ", train_targets.shape)
    # print("test_targets.shape: ", test_targets.shape)
    
    # print("\n \n \n")
    # print("Features: ", features)
    # print("train_features: ", train_features)
    # print("test_features: ", test_features)
    # print("\n \n \n")
    # print("targets: ", targets)
    # print("train_targets: ", train_targets)
    # print("test_targets: ", test_targets)

    return train_features, train_targets, test_features, test_targets

    #raise NotImplementedError


def cross_validation(features, targets, folds):
    """
    Split the data in `folds` different groups for cross-validation.
        Split the features and targets into a `folds` number of groups that
        divide the data as evenly as possible. Then for each group,
        return a tuple that treats that group as the test set and all
        other groups combine to make the training set.

        Note that this should be *deterministic*; don't shuffle the data.
        If there are 100 examples and you have 5 folds, each group
        should contain 20 examples and the first group should contain
        the first 20 examples.

        See test_cross_validation for expected behavior.

    Args:
        features: an NxK matrix of N examples, each with K featuress
        targets: an Nx1 array of N labels
        folds: the number of cross-validation groups

    Output:
        A list of tuples, where each tuple contains:
          (train_features, train_targets, test_features, test_targets)
    """

    print('features = ', features)
    print('targets =', targets)
    print('folds =', folds)



    test_size = int((np.shape(targets)[0])/folds)
    print("\n \n")
    print("test_size = ", test_size)

    train_features = np.array([])
    train_targets = np.array([])
    test_features = np.array([])
    test_targets = np.array([])

    cross_validation_list_of_tuples = []
    cross_validation_tuple = ()


    for i in range(0,folds):
        print("\n \n")
        print("i = ", i)

        train_features = features[test_size:,:]   
        test_features = features[0:test_size,:]
        train_targets = targets[test_size:,:]
        test_targets = targets[0:test_size,:]  

        cross_validation_tuple = (train_features,train_targets,test_features,test_targets)
        cross_validation_list_of_tuples.append(cross_validation_tuple) # Create a list of tuples
        
        features = np.concatenate((train_features,test_features),axis = 0)
        targets = np.concatenate((train_targets,test_targets),axis = 0)


        print("train_features = ", train_features)
        print("train_targets = ", train_targets)
        print("test_features = ", test_features)
        print("test_targets = ", test_targets)

        print("cross_validation_tuple = ", cross_validation_list_of_tuples)



    assert features.shape[0] == targets.shape[0]

    if folds == 1:
        return [(features, targets, features, targets)]

    return cross_validation_list_of_tuples
    #raise NotImplementedError
