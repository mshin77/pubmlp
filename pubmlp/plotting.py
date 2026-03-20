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
                label=f'Best Validation Loss: {best_val_loss:.3f}')
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


def plot_al_progress(al_history, criteria=None, x_col='n_coded',
                     show_per_label=False, save_path=None):
    """
    Plot active learning progress as a learning curve.

    Single plot showing Macro F1 vs number of labeled instances.
    For multi-label tasks, optionally overlays per-criterion F1 lines.

    Args:
        al_history: list of dicts or DataFrame with columns including
            ``macro_f1`` and ``{criterion}_f1`` per iteration.
        criteria: list of criterion names (auto-detected from ``_f1``
            columns if None).
        x_col: column for x-axis (default 'n_coded').
        show_per_label: if True, overlay per-criterion F1 as thin lines.
            Ignored when there is only one criterion.
        save_path: path to save figure (optional).
    """
    import pandas as pd
    al_df = pd.DataFrame(al_history) if not isinstance(al_history, pd.DataFrame) else al_history

    if len(al_df) < 2:
        print("Need at least 2 iterations to plot AL progress.")
        return

    if criteria is None:
        criteria = [c.replace('_f1', '') for c in al_df.columns if c.endswith('_f1')]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(al_df[x_col], al_df['macro_f1'], 'o-', linewidth=2.5,
            markersize=6, color='#1f77b4', label='Macro F1', zorder=3)

    if show_per_label and len(criteria) > 1:
        for c in criteria:
            if f'{c}_f1' in al_df.columns:
                ax.plot(al_df[x_col], al_df[f'{c}_f1'], '--',
                        linewidth=1, markersize=3, alpha=0.6, label=c)

    ax.set_xlabel('Number of Labeled Instances', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Active Learning Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
