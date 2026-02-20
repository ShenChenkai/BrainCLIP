import numpy as np
import nni
import torch
import torch.nn.functional as F
from sklearn import metrics
from typing import Optional
from torch.utils.data import DataLoader
import logging
from src.utils import mixup, mixup_criterion


def train_and_evaluate(model, train_loader, test_loader, optimizer, device, args):
    model.train()
    accs, aucs, macros = [], [], []
    epoch_losses = []
    val_losses = []
    val_epochs = []
    epoch_num = args.epochs

    # Add scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    for i in range(epoch_num):
        loss_all = 0
        for data in train_loader:
            data = data.to(device)

            if args.mixup:
                data, y_a, y_b, lam = mixup(data)
            optimizer.zero_grad()
            out = model(data)

            if args.mixup:
                loss = mixup_criterion(F.nll_loss, out, y_a, y_b, lam)
            else:
                loss = F.nll_loss(out, data.y)

            loss.backward()
            optimizer.step()

            loss_all += loss.item() * data.num_graphs
        epoch_loss = loss_all / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)

        # train_micro, train_auc, train_macro = evaluate(model, device, train_loader)
        logging.info(f'(Train) | Epoch={i:03d}, loss={epoch_loss:.4f}') #, '
                     # f'train_micro={(train_micro * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
                     # f'train_auc={(train_auc * 100):.2f}')

        if (i + 1) % args.test_interval == 0:
            test_micro, test_auc, test_macro, test_loss = evaluate(model, device, test_loader)
            accs.append(test_micro)
            aucs.append(test_auc)
            macros.append(test_macro)
            val_losses.append(test_loss)
            val_epochs.append(i + 1)

            text = f'(Train Epoch {i}), test_micro={(test_micro * 100):.2f}, ' \
                   f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}, ' \
                   f'test_loss={test_loss:.4f}\n'
            logging.info(text)

        if args.enable_nni:
            # Evaluate on training set to get train_auc for NNI reporting
            train_micro, train_auc, train_macro, train_loss = evaluate(model, device, train_loader)
            nni.report_intermediate_result(train_auc)

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
        if len(val_losses) > 0:
            plt.plot(val_epochs, val_losses, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_curve.png')
        plt.close()
        logging.info('Loss curve saved to loss_curve.png')
    except ImportError:
        logging.warning('Matplotlib not installed. Skipping loss plot.')

    try:
        data_to_save = np.column_stack((range(1, len(epoch_losses) + 1), epoch_losses))
        header = 'epoch,train_loss'
        if len(val_losses) > 0:
             # Extend val_losses to match epoch length directly or just save separate file?
             # Saving separate file is cleaner.
             np.savetxt('val_loss_history.csv', np.column_stack((val_epochs, val_losses)), delimiter=',', header='epoch,val_loss', comments='')
        np.savetxt('train_loss_history.csv', data_to_save, delimiter=',', header=header, comments='')
        logging.info('Loss histories saved to .csv files')
    except Exception as e:
        logging.warning(f'Failed to save loss history: {e}')

    accs, aucs, macros = np.sort(np.array(accs)), np.sort(np.array(aucs)), np.sort(np.array(macros))
    return accs.mean(), aucs.mean(), macros.mean(), (np.mean(val_losses) if val_losses else 0.0)


@torch.no_grad()
def evaluate(model, device, loader, test_loader: Optional[DataLoader] = None):
    model.eval()
    preds, trues, preds_prob = [], [], []
    loss_all = 0
    total_samples = 0

    for data in loader:
        data = data.to(device)
        c = model(data)
        
        loss = F.nll_loss(c, data.y, reduction='sum')
        loss_all += loss.item()
        total_samples += data.y.size(0)

        pred = c.max(dim=1)[1]
        preds += pred.detach().cpu().tolist()
        preds_prob += torch.exp(c)[:, 1].detach().cpu().tolist()
        trues += data.y.detach().cpu().tolist()

    val_loss = loss_all / total_samples

    try:
        train_auc = metrics.roc_auc_score(trues, preds_prob)
    except ValueError:
        train_auc = 0.5

    train_micro = metrics.f1_score(trues, preds, average='micro')
    train_macro = metrics.f1_score(trues, preds, average='macro', labels=[0, 1])

    if test_loader is not None:
        test_micro, test_auc, test_macro, test_loss = evaluate(model, device, test_loader)
        return train_micro, train_auc, train_macro, val_loss, test_micro, test_auc, test_macro, test_loss
    else:
        return train_micro, train_auc, train_macro, val_loss
