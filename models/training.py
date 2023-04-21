import torch
from tqdm.auto import tqdm
import numpy as np

def train_loop(dataloader, model, optimizer, device, local_device, skip_visual=False):
    print("Training...")
    model.train()
    progress_bar = tqdm(range(len(dataloader)), ascii=True, disable=skip_visual)
    losses = []
    for XY in dataloader:
        XY = [xy.to(device) for xy in XY]
        Xs = XY[:-1]
        Y = XY[-1]
        pred = model(*Xs)
        loss = model.compute_loss(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().to(local_device).numpy())
        progress_bar.update(1)
    print('Train loss: ' + str(np.mean(losses)))


def eval_loop(dataloader, model, device, local_device, skip_visual=False):
    print("Evaluating...")
    model.eval()
    progress_bar = tqdm(range(len(dataloader)), ascii=True, disable=skip_visual)
    correct = 0
    size = 0
    TPs = 0
    FPs = 0
    FNs = 0
    preds = []
    with torch.no_grad():
        for XY in dataloader:
            XY = [xy.to(device) for xy in XY]
            Xs = XY[:-1]
            Y = XY[-1].to(local_device)
            pred = model.postprocessing(model(*Xs))
            preds.append(pred)
            Y = Y.numpy()
            eq = np.equal(Y, pred)
            size += len(eq)
            correct += sum(eq)
            TPs += sum(np.logical_and(np.equal(Y, 1.0), np.equal(pred, 1.0)))
            FPs += sum(np.logical_and(np.equal(Y, 0.0), np.equal(pred, 1.0)))
            FNs += sum(np.logical_and(np.equal(Y, 1.0), np.equal(pred, 0.0)))
            progress_bar.update(1)
    print('Accuracy: ' + str(correct / size))
    # print('Binary F1: ' + str(2 * TPs / (2 * TPs + FPs + FNs)))
    preds = np.concatenate(preds)
    return (preds)
