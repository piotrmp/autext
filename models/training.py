import torch
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import numpy as np


def train_loop(dataloader, model, optimizer, scheduler, device, local_device, skip_visual=False):
    print("Training...")
    model.train()
    progress_bar = tqdm(range(len(dataloader)), ascii=True, disable=skip_visual)
    losses = []
    preds = []
    true_Y = []
    for XY in dataloader:
        XY = [xy.to(device) for xy in XY]
        Xs = XY[:-1]
        Y = XY[-1]
        raw_pred = model(*Xs)
        preds.append(model.postprocessing(raw_pred))
        true_Y.append(Y.to(local_device).numpy())
        loss = model.compute_loss(raw_pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().to(local_device).numpy())
        progress_bar.update(1)
    scheduler.step()
    print('Train loss: ' + str(np.mean(losses)))
    preds = np.concatenate(preds)
    true_Y = np.concatenate(true_Y)
    f1 = f1_score(y_true=true_Y, y_pred=preds, average="macro")
    return f1


def eval_loop(dataloader, model, device, local_device, skip_visual=False, test=False):
    if test:
        print("Generating predictions...")
    else:
        print("Evaluating...")
    model.eval()
    progress_bar = tqdm(range(len(dataloader)), ascii=True, disable=skip_visual)
    correct = 0
    size = 0
    preds = []
    true_Y = []
    with torch.no_grad():
        for XY in dataloader:
            XY = [xy.to(device) for xy in XY]
            Xs = XY[:-1]
            Y = XY[-1].to(local_device)
            pred = model.postprocessing(model(*Xs))
            preds.append(pred)
            Y = Y.numpy()
            true_Y.append(Y)
            eq = np.equal(Y, pred)
            size += len(eq)
            correct += sum(eq)
            progress_bar.update(1)
    preds = np.concatenate(preds)
    true_Y = np.concatenate(true_Y)
    if not test:
        print('Accuracy: ' + str(correct / size))
        f1 = f1_score(y_true=true_Y, y_pred=preds, average="macro")
        print('F1 score: ' + str(f1))
        return f1
    else:
        return preds
