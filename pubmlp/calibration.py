import torch
import torch.nn as nn
import torch.optim as optim

from .utils import default_forward_fn


def collect_logits(model, dataloader, device, forward_fn=None):
    """Run model in eval mode and collect raw logits + labels."""
    if forward_fn is None:
        forward_fn = default_forward_fn
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels'].to(device)
            logits = forward_fn(model, batch, device)
            all_logits.append(logits)
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


class TemperatureScaling:
    """Per-label temperature scaling. Works for single and multi-label."""

    def __init__(self):
        self.temperature = None

    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """Optimize temperature via NLL loss using LBFGS."""
        num_labels = logits.shape[-1] if logits.dim() > 1 else 1

        if num_labels == 1:
            logits_flat = logits.view(-1)
            labels_flat = labels.view(-1).float()
            log_temp = nn.Parameter(torch.zeros(1, device=logits.device))
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

            def closure():
                optimizer.zero_grad()
                loss = criterion(logits_flat / log_temp.exp(), labels_flat)
                loss.backward()
                return loss

            optimizer.step(closure)
            self.temperature = log_temp.exp().item()
        else:
            # Per-label temperature
            temps = []
            criterion = nn.BCEWithLogitsLoss()
            for i in range(num_labels):
                log_temp = nn.Parameter(torch.zeros(1, device=logits.device))
                optimizer = optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)
                col_logits = logits[:, i]
                col_labels = labels[:, i].float()

                def closure(lt=log_temp, l=col_logits, lb=col_labels, opt=optimizer):
                    opt.zero_grad()
                    loss = criterion(l / lt.exp(), lb)
                    loss.backward()
                    return loss

                optimizer.step(closure)
                temps.append(log_temp.exp().item())
            self.temperature = temps
        return self

    def transform(self, logits):
        if isinstance(self.temperature, list):
            temp_tensor = torch.tensor(self.temperature, device=logits.device).unsqueeze(0)
            return logits / temp_tensor
        return logits / self.temperature

    def to_dict(self):
        return {'temperature': self.temperature}

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.temperature = d['temperature']
        return obj


def calibrate_model(model, dataloader, device, forward_fn=None):
    """Collect logits and fit temperature scaling. Pass ``cached_forward_fn`` for cached batches."""
    logits, labels = collect_logits(model, dataloader, device, forward_fn=forward_fn)
    scaler = TemperatureScaling()
    scaler.fit(logits, labels)
    return scaler
