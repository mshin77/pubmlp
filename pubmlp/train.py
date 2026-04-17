import time
import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from .utils import default_forward_fn


def calculate_loss(model, dataloader, criterion, device, forward_fn=None):
    """Average loss across all batches."""
    if forward_fn is None:
        forward_fn = default_forward_fn
    model.eval()
    total_loss, n_samples = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels'].to(device)
            outputs = forward_fn(model, batch, device)
            loss = criterion(outputs, labels)
            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            n_samples += batch_n
    return total_loss / n_samples


def calculate_accuracy(model, dataloader, device, forward_fn=None):
    """Accuracy (%) across all batches. Multi-label: average per-label accuracy."""
    if forward_fn is None:
        forward_fn = default_forward_fn
    model.eval()
    correct, total_elements = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels'].to(device)
            outputs = forward_fn(model, batch, device)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total_elements += labels.numel()
    return correct / total_elements * 100


def calculate_pos_weight(dataloader, device):
    """Compute pos_weight from label distribution: neg_count / pos_count per label."""
    all_labels = []
    for batch in dataloader:
        all_labels.append(batch['labels'])
    labels = torch.cat(all_labels, dim=0).float()
    pos_count = labels.sum(dim=0)
    neg_count = labels.shape[0] - pos_count
    return (neg_count / pos_count.clamp(min=1)).to(device)


def train_evaluate_model(model, train_dataloader, validation_dataloader, test_dataloader,
                         optimizer, criterion, device, epochs, scheduler=None,
                         early_stopping_patience=3, use_best_model=True,
                         gradient_clip_norm=1.0, use_warmup=True, pos_weight='auto',
                         warmup_steps=0, forward_fn=None):
    """
    Train model with early stopping, evaluate on test set.

    Args:
        pos_weight: 'auto' computes from training labels, None disables,
                    or pass a tensor directly.
        warmup_steps: Number of warmup steps for linear schedule. 0 = no warmup.
        test_dataloader: Optional. If None, test evaluation is skipped.
        forward_fn: Forward override; pass ``cached_forward_fn`` for cached-embedding batches.

    Returns:
        tuple: (train_losses, validation_losses, train_accuracies,
                validation_accuracies, test_accuracy, best_val_loss,
                best_model_state, best_epoch)
    """
    start_time = time.time()
    model.to(device)

    if forward_fn is None:
        forward_fn = default_forward_fn

    # Handle pos_weight for BCEWithLogitsLoss
    if pos_weight == 'auto':
        computed_pos_weight = calculate_pos_weight(train_dataloader, device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=computed_pos_weight)
    elif pos_weight is not None:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_losses, validation_losses = [], []
    train_accuracies, validation_accuracies = [], []
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    if use_warmup and scheduler is None:
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        total_loss, n_samples = 0.0, 0

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            labels = batch['labels'].to(device)
            outputs = forward_fn(model, batch, device)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()

            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            n_samples += batch_n
            progress_bar.set_postfix({'loss': total_loss / n_samples})

        train_loss = total_loss / n_samples
        train_losses.append(train_loss)

        validation_loss = calculate_loss(model, validation_dataloader, criterion, device, forward_fn=forward_fn)
        validation_losses.append(validation_loss)

        train_accuracy = calculate_accuracy(model, train_dataloader, device, forward_fn=forward_fn)
        train_accuracies.append(train_accuracy)

        validation_accuracy = calculate_accuracy(model, validation_dataloader, device, forward_fn=forward_fn)
        validation_accuracies.append(validation_accuracy)

        improved = validation_loss < best_val_loss
        if improved:
            best_val_loss = validation_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
            print(f'Epoch: {epoch+1:04d}/{epochs:04d} | Training Loss: {train_loss:.3f} | Validation Loss: {validation_loss:.3f} *** Best ***')
        else:
            patience_counter += 1
            print(f'Epoch: {epoch+1:04d}/{epochs:04d} | Training Loss: {train_loss:.3f} | Validation Loss: {validation_loss:.3f}')

        print(f'Training Accuracy: {train_accuracy:.3f}% | Validation Accuracy: {validation_accuracy:.3f}% | {(time.time() - start_time)/60:.2f} min')

        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(validation_loss)

        if patience_counter >= early_stopping_patience:
            print(f'\nEarly stopping after {epoch+1} epochs (no improvement for {early_stopping_patience} epochs)')
            break

    if use_best_model and best_model_state is not None:
        print(f'\nLoading best model from epoch {best_epoch}')
        model.load_state_dict(best_model_state)

    if test_dataloader is not None:
        test_accuracy = calculate_accuracy(model, test_dataloader, device, forward_fn=forward_fn)
        print(f'Test Accuracy: {test_accuracy:.3f}% | Best epoch {best_epoch} (validation loss: {best_val_loss:.3f})')
    else:
        test_accuracy = None
        print(f'Best epoch {best_epoch} (validation loss: {best_val_loss:.3f})')

    return (train_losses, validation_losses, train_accuracies, validation_accuracies,
            test_accuracy, best_val_loss, best_model_state, best_epoch)
