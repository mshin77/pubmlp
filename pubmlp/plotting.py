import matplotlib.pyplot as plt


def plot_results(train_losses, validation_losses, train_accuracies, validation_accuracies,
                 test_accuracy, best_val_loss, best_epoch=None):
    """Plot training/validation loss and accuracy curves."""
    if len(train_losses) != len(validation_losses) or len(train_accuracies) != len(validation_accuracies):
        raise ValueError("Input lists must have the same length")

    num_epochs = len(train_losses)
    epochs = list(range(1, num_epochs + 1))
    best_x = best_epoch if best_epoch is not None else num_epochs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, linewidth=2, label='Training Loss')
    ax1.plot(epochs, validation_losses, linewidth=2, label='Validation Loss')
    ax1.scatter(best_x, best_val_loss, color='red', marker='o', s=80, zorder=5,
                label=f'Best Val Loss: {best_val_loss:.3f}')
    ax1.set_title('Loss', fontsize=15, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Loss', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.tick_params(labelsize=12)

    ax2.plot(epochs, train_accuracies, linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, validation_accuracies, linewidth=2, label='Validation Accuracy')
    if test_accuracy is not None:
        ax2.scatter(num_epochs, test_accuracy, color='blue', marker='o', s=80, zorder=5,
                    label=f'Test Accuracy: {test_accuracy:.3f}')
    ax2.set_title('Accuracy', fontsize=15, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=13)
    ax2.set_ylabel('Accuracy', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.tick_params(labelsize=12)

    fig.tight_layout()
    plt.show()
