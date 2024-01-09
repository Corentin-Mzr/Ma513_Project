import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from math import ceil


def visualize_data(x: np.ndarray, y: np.ndarray, indexes: list[int] | np.ndarray) -> None:
    """
    Plot the images and their corresponding labels
    :param x: List of images
    :param y: List of labels
    :param indexes: List of indexes
    """
    n = int(ceil(len(indexes)**0.5))
    fig, ax = plt.subplots(nrows=n, ncols=n)
    i = 0
    # Multiple plots
    try:
        ax = ax.flatten()
        for idx in indexes:
            ax[i].imshow(x[idx].reshape(32, 32), cmap='gray')
            ax[i].set_title(f'Label : {y[idx]}')
            ax[i].axis('off')
            i += 1
    # Only one plot
    except AttributeError:
        i = indexes[0]
        ax.imshow(x[i].reshape(32, 32), cmap='gray')
        ax.set_title(f'Label : {y[i]}')
        ax.axis('off')
    finally:
        # Remove unused subplots
        nb_unused = n ** 2 - len(indexes)
        for i in range(n ** 2 - 1, n ** 2 - nb_unused - 1, -1):
            fig.delaxes(fig.axes[i])
        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plot the confusion matrix
    :param y_true: True labels
    :param y_pred: Predicted labels
    """
    plt.imshow(confusion_matrix(y_true, y_pred), cmap='Purples')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.colorbar()
    plt.show()


def plot_model_training(hist: dict, epochs: int) -> None:
    """
    Plot the evolution of loss and accuracy for a model
    :param hist: Model's history
    :param epochs: Number of epochs
    """
    # Extract loss and accuracy from history
    loss = hist['loss']
    acc = hist['accuracy']
    val_loss = hist['val_loss']
    val_acc = hist['val_accuracy']

    # Time
    t = np.arange(0, epochs, 1)

    # Plot the loss and accuracy
    fig, ax = plt.subplots(ncols=2)
    ax = ax.flatten()

    # Plot loss
    ax[0].plot(t, loss, c='blue', label='Loss')
    ax[0].plot(t, val_loss, c='red', label='Validation Loss')
    ax[0].set_title('Loss per epoch')
    ax[0].legend()

    # Plot accuracy
    ax[1].plot(t, acc, c='blue', label='Accuracy')
    ax[1].plot(t, val_acc, c='red', label='Validation Accuracy')
    ax[1].set_title('Accuracy per epoch')
    ax[1].legend()

    plt.show()


def print_scores(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Print the accuracy, F1 score, recall and precision
    :param y_true: True labels
    :param y_pred: Predicted labels
    """
    # Compute accuracy
    res = np.zeros(len(y_true))
    res[y_true == y_pred] = 1
    accuracy = 100 * sum(res) / len(res)

    # Get F1 score,recall and precision from classification_report
    report: dict = classification_report(y_true, y_pred, output_dict=True)['weighted avg']
    precision = report['precision']
    f1 = report['f1-score']
    recall = report['recall']

    # Print scores
    print(f"Accuracy : {accuracy:.2f}%\n"
          f"F1 score : {f1:.2f}\n"
          f"Precision : {precision:.2f}\n"
          f"Recall : {recall:.2f}")
