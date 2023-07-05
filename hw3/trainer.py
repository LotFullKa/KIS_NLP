from tqdm.auto import tqdm

import torch

from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from accelerate import Accelerator
import evaluate

import numpy as np


TICKS_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 16

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_masked_word_logits(logits, masked_index):
    return torch.diagonal(logits[:, masked_index, :]).T

def train_epoch(model, optimizer, train_loader, lr_scheduler, accelerator):
    model.train()

    losses = []
    for i, batch in enumerate(train_loader):
        tokens, masked_word, masked_index = batch
        
        logits = model(**tokens).logits
        loss = cross_entropy(get_masked_word_logits(logits, masked_index), masked_word)

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        optimizer.zero_grad()

        losses.append(loss.detach().cpu())
    
    return np.mean(losses)


def eval_model(model, metric, data_loader):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            tokens, masked_word, masked_index = batch
            logits = model(**tokens).logits
            predictions = get_masked_word_logits(logits, masked_index).argmax(-1)
            # print(predictions.shape, masked_word.shape)
            metric.add_batch(predictions=predictions.detach().cpu().numpy(), references=masked_word.detach().cpu().numpy())


def train_model(
    model,
    train_loader,
    test_loader,
    epochs,
    lr,
    use_tqdm=False,
):
    accelerator = Accelerator()
    optimizer = Adam(model.parameters(), lr=lr/25)
    lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
    metric = evaluate.load('accuracy')

    model, optimizer, train_loader, test_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, lr_scheduler
    )

    forrange = tqdm(range(epochs)) if use_tqdm else range(epochs)
    
    model = model.to(device)

    train_loss = []
    for epoch in forrange:
        model.train()
        mean_epoch_loss = train_epoch(model, optimizer, train_loader, lr_scheduler, accelerator)
        eval_model(model, metric, test_loader)

        train_loss.append(mean_epoch_loss)
        eval_metric = metric.compute()
        # Use accelerator.print to print only on the main process.
        print(f"epoch {epoch}:")
        for k, v in eval_metric.items():
            print(f"{k}: {v}")
    return train_loss