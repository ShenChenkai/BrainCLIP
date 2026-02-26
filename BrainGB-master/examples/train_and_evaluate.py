import numpy as np
import nni
import torch
import torch.nn.functional as F
from sklearn import metrics
from typing import Optional
from torch.utils.data import DataLoader
import logging
from src.utils import mixup, mixup_criterion


class WarmupCosineLRScheduler:
    def __init__(self, optimizer, total_epochs, warmup_epochs=10, min_lr_ratio=0.01, warmup_start_factor=0.1):
        self.optimizer = optimizer
        self.total_epochs = max(1, int(total_epochs))
        self.warmup_epochs = int(max(0, min(warmup_epochs, self.total_epochs - 1)))
        self.min_lr_ratio = float(max(0.0, min(1.0, min_lr_ratio)))
        self.warmup_start_factor = float(max(1e-8, min(1.0, warmup_start_factor)))
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0

        if self.warmup_epochs > 0:
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * self.warmup_start_factor

    def _get_factor(self, epoch_idx):
        if self.warmup_epochs > 0 and epoch_idx <= self.warmup_epochs:
            if self.warmup_epochs == 1:
                return 1.0
            alpha = (epoch_idx - 1) / (self.warmup_epochs - 1)
            return self.warmup_start_factor + alpha * (1.0 - self.warmup_start_factor)

        decay_epochs = self.total_epochs - self.warmup_epochs
        if decay_epochs <= 0:
            return 1.0

        progress = (epoch_idx - self.warmup_epochs) / decay_epochs
        progress = float(max(0.0, min(1.0, progress)))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self):
        self.current_epoch += 1
        factor = self._get_factor(self.current_epoch)
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * factor


def build_lr_scheduler(optimizer, args):
    scheduler_name = str(getattr(args, 'lr_scheduler', 'warmup_cosine')).lower()

    if scheduler_name == 'none':
        return None, scheduler_name

    if scheduler_name == 'reduce_on_plateau':
        factor = float(getattr(args, 'plateau_factor', 0.5))
        patience = int(getattr(args, 'plateau_patience', 10))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
        )
        return scheduler, scheduler_name

    total_epochs = max(1, int(getattr(args, 'epochs', 1)))
    warmup_epochs = int(getattr(args, 'warmup_epochs', 10))
    warmup_epochs = max(0, min(warmup_epochs, total_epochs - 1))

    min_lr_ratio = float(getattr(args, 'min_lr_ratio', 0.01))
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))
    warmup_start_factor = float(getattr(args, 'warmup_start_factor', 0.1))
    scheduler = WarmupCosineLRScheduler(
        optimizer=optimizer,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        min_lr_ratio=min_lr_ratio,
        warmup_start_factor=warmup_start_factor,
    )

    return scheduler, 'warmup_cosine'


def train_and_evaluate(model, train_loader, test_loader, optimizer, device, args):
    model.train()
    accs, aucs, macros = [], [], []
    epoch_losses = []
    val_losses = []
    val_epochs = []
    epoch_num = args.epochs

    scheduler, scheduler_name = build_lr_scheduler(optimizer, args)

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

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'(Train) | Epoch={i:03d}, lr={current_lr:.6e}, loss={epoch_loss:.4f}') #, '
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

        if scheduler is not None:
            if scheduler_name == 'reduce_on_plateau':
                monitor_loss = val_losses[-1] if len(val_losses) > 0 else epoch_loss
                scheduler.step(monitor_loss)
            else:
                scheduler.step()

        if args.enable_nni:
            # Evaluate on training set to get train_auc for NNI reporting
            train_micro, train_auc, train_macro, train_loss = evaluate(model, device, train_loader)
            nni.report_intermediate_result(train_auc)

    accs, aucs, macros = np.sort(np.array(accs)), np.sort(np.array(aucs)), np.sort(np.array(macros))
    loss_history = {
        'train_losses': epoch_losses,
        'val_losses': val_losses,
        'val_epochs': val_epochs,
    }
    return accs.mean(), aucs.mean(), macros.mean(), (np.mean(val_losses) if val_losses else 0.0), loss_history


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
