import matplotlib.pyplot as plt


def plot_results(train_losses, validation_losses, train_accuracies, validation_accuracies,
                 test_accuracy, best_val_loss, best_epoch=None):
    """Plot training/validation loss and accuracy curves."""
    if len(train_losses) != len(validation_losses) or len(train_accuracies) != len(validation_accuracies):
        raise ValueError("Input lists must have the same length")

    num_epochs = len(train_losses)
    epochs = list(range(1, num_epochs + 1))
    best_x = best_epoch if best_epoch is not None else num_epochs

    plt.figure(figsize=(9, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.scatter(best_x, best_val_loss, color='red', marker='o', s=50,
                label=f'Best Val Loss: {best_val_loss:.3f}')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy')
    if test_accuracy is not None:
        plt.scatter(num_epochs, test_accuracy, color='blue', marker='o', s=50,
                    label=f'Test Accuracy: {test_accuracy:.3f}')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
