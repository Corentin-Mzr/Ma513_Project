import numpy as np
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, LeakyReLU, Reshape, Flatten, ReLU
from keras import Input, Model
from keras.optimizers import Adam
from sklearn.svm import LinearSVC
from data_related import save_data


def cnn_model() -> keras.Model:
    """
    CNN Model as described in the paper
    """
    # Input reshaped
    inp = Input(shape=(1024, 1), name='Input')
    x = Reshape(target_shape=(32, 32, 1), name='Reshape')(inp)

    # First part
    x = Conv2D(filters=36, kernel_size=(5, 5), strides=1, padding='same', name='Conv1')(x)
    x = LeakyReLU(alpha=0.01, name='Leaky1')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same', name='MaxPooling1')(x)

    # Second part
    x = Conv2D(filters=72, kernel_size=(5, 5), strides=1, padding='same', name='Conv2')(x)
    x = LeakyReLU(alpha=0.01, name='Leaky2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same', name='MaxPooling2')(x)

    # Transition from 2D to 1D
    x = Flatten()(x)

    # Third part
    x = Dense(units=1024, name='Dense3')(x)
    x = LeakyReLU(alpha=0.01, name='Leaky3')(x)
    x = Dropout(rate=0.85, name='Dropout3')(x)

    # Output -> we extract 25 features that we will put into an SVM classifier
    output = Dense(units=25, activation='softmax', name='Output')(x)

    return Model(inputs=inp, outputs=output)


def my_cnn_model() -> keras.Model:
    """
    My CNN model, a modified version of the one presented in the paper
    """
    # Input reshaped
    inp = Input(shape=(1024, 1), name='Input')
    x = Reshape(target_shape=(32, 32, 1), name='Reshape')(inp)

    # First part
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', name='Conv1')(x)
    x = ReLU(name='ReLU1')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='MaxPooling1')(x)
    x = Dropout(0.2)(x)

    # Second part
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', name='Conv2')(x)
    x = ReLU(name='ReLU2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='MaxPooling2')(x)
    x = Dropout(0.1)(x)

    # Second part
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', name='Conv3')(x)
    x = ReLU(name='ReLU3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='MaxPooling3')(x)
    x = Dropout(0.2)(x)

    # Second part
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', name='Conv5')(x)
    x = ReLU(name='ReLU5')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='MaxPooling5')(x)
    x = Dropout(0.1)(x)

    # Second part
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', name='Conv6')(x)
    x = ReLU(name='ReLU6')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='MaxPooling6')(x)
    x = Dropout(0.4)(x)

    # Transition from 2D to 1D
    x = Flatten()(x)

    # Third part
    x = Dense(units=128, name='Dense3')(x)
    x = ReLU(name='ReLU0')(x)

    # Output -> we extract 25 features that we will put into an SVM classifier
    output = Dense(units=25, activation='softmax', name='Output')(x)

    return Model(inputs=inp, outputs=output)


def svm_classifier(x_train: np.ndarray, y_train: np.ndarray, cnn: keras.Model = None) -> LinearSVC:
    """
    Build the SVM classifier
    :param x_train: Training features set
    :param y_train: Training labels set
    :param cnn: CNN Model, if None builds the SVM from x_train instead of x_features
    :return: SVM Classifier
    """
    x_features = x_train
    # Feature extraction using the CNN Model
    if cnn is not None:
        x_features = cnn.predict(x_train)

    # Create and train SVM
    classifier = LinearSVC(penalty='l2',
                           C=10.0,
                           multi_class='ovr',
                           loss='squared_hinge',
                           dual='auto',
                           max_iter=1000,
                           tol=1e-4)
    classifier.fit(x_features, y_train)

    return classifier


def train(model: keras.Model,
          x_train: np.ndarray,
          y_train: np.ndarray,
          epochs: int,
          batch_size: int,
          validation_split: float,
          learning_rate: float,
          autosave: bool = False,
          model_save_file: str = "models/my_model.keras",
          history_save_file: str = "models/my_history.pkl") -> tuple[keras.Model, dict]:
    """
    Train the model with given parameters
    :param model: Model
    :param x_train: Training features
    :param y_train: Training labels
    :param epochs: Number of epochs
    :param batch_size: Size of batch
    :param validation_split: Validation split
    :param learning_rate: Adam learning rate
    :param autosave: True for autosave
    :param model_save_file: Name of the file for model saving
    :param history_save_file: Name of the file for history saving
    :return: The model and history
    """
    # Compile the model
    model.compile(optimizer=Adam(learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Train the model
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=validation_split,
              workers=8,
              use_multiprocessing=True)

    hist = model.history.history

    if not autosave:
        to_save = input('Save model and history ? [y/N]')
    else:
        to_save = 'y'

    if to_save.lower() == 'y':
        model.save(model_save_file)
        save_data(hist, history_save_file)

    return model, hist
