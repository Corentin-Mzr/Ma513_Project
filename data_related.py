import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def load_data(data_path: str, load_pickle: bool = False) -> np.ndarray | dict:
    """
    Load and retrieve data from a npz file or a pickle file
    :param data_path: Path to the file
    :param load_pickle: if True, file is a pickle file and the content is a dictionary
    :return: Numpy array or dictionary containing the data
    """
    print(f"Loading {data_path}")
    if load_pickle:
        with open(data_path, 'rb') as file:
            d = pickle.load(file)
        file.close()
        return d
    else:
        data = np.load(data_path, allow_pickle=True)
        array = data['arr']
        data.close()
        return array


def separate_data(data_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Create two numpy arrays, the first one containing the images and the second one containing the labels
    Images are 32x32 numpy arrays
    Labels are going from 0 to 24
    :param data_array: Numpy array loaded with load_data
    :return: Tuple of numpy arrays
    """
    # x contains all malware data as images of shape 32x32
    # we will reshape it as a matrix with 32*32 = 1024 columns
    x = np.array([arr.reshape(32 * 32) for arr in data_array[:, 0]]).reshape((-1, 32 * 32))

    # y contains the class of each malware, labelled from 0 to 24
    y = to_categorical(np.array(data_array[:, 1], dtype='int'), num_classes=25)

    return x, y


def normalize_data(data_array: np.ndarray) -> np.ndarray:
    """
    Normalize the data, each value will be between 0 and 1
    :param data_array: Data to normalize
    :return: Normalized data
    """
    return data_array / np.max(data_array)


def split_data(x: np.ndarray, y: np.ndarray, seed: int = None) -> tuple:
    """
    Create test and train sets
    :param x: Features
    :param y: Labels
    :param seed: Chosen seed for reproducibility
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed if seed is not None else None)
    return x_train, x_test, y_train, y_test


def get_all_data(path: str = 'datasets/images_malware.npz', seed: int = None) -> dict:
    # Load data
    data = load_data(path)

    # Separate data
    x, y = separate_data(data)

    # Normalize data
    norm_x, norm_y = normalize_data(x), normalize_data(y)

    # Split data
    x_train, x_test, y_train, y_test = split_data(norm_x, norm_y, seed)

    # Create a dictionary containing the data in all forms
    d = {'x': x,
         'y': y,
         'x_train': x_train,
         'x_test': x_test,
         'y_train': y_train,
         'y_test': y_test}

    return d


def save_data(data: dict, path: str = 'datasets/dict_data.pkl') -> None:
    """
    Save the dictionary data to a pickle file
    :param data: Dictionary containing all the datasets
    :param path: Path to the file
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {path}")
    f.close()
