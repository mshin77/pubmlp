"""Precompute encoder embeddings for the cached-embedding fast path."""

import torch


@torch.no_grad()
def compute_cls_embeddings(model, dataloader, device, show_progress=True):
    """Run the encoder once over a dataloader; return (N, hidden_size) CPU tensor."""
    was_training = model.training
    model.eval()
    try:
        iterator = dataloader
        if show_progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(dataloader, desc='Computing embeddings')
            except ImportError:
                pass

        embeddings = []
        for batch in iterator:
            if model._use_sentence_transformer:
                texts = batch.get('texts')
                if texts is None:
                    raise ValueError(
                        "Batch is missing 'texts'; sentence-transformer encoder needs raw texts."
                    )
                emb = model._encode(None, None, texts=texts)
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                emb = model._encode(input_ids, attention_mask)
            embeddings.append(emb.detach().to('cpu'))
        return torch.cat(embeddings, dim=0)
    finally:
        if was_training:
            model.train()
