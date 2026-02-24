import numpy as np
import torch
import logging
from src.models.losses import info_nce_loss


def pretrain(model, train_loader, optimizer, device, args):
    model.train()
    epoch_losses = []
    epoch_num = args.epochs

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    for i in range(epoch_num):
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            v_graph, v_text = model(
                data.x, data.edge_index, data.edge_attr, data.batch,
                data.input_ids, data.attention_mask,
            )
            if i == 0 and loss_all == 0:  # print once: first batch of first epoch
                print(f"[DEBUG] v_graph.shape={v_graph.shape}, v_text.shape={v_text.shape}")
            loss = info_nce_loss(v_graph, v_text)

            loss.backward()
            optimizer.step()

            loss_all += loss.item() * data.num_graphs
        epoch_loss = loss_all / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        scheduler.step(epoch_loss)

        logging.info(f'(Pretrain) | Epoch={i:03d}, loss={epoch_loss:.4f}')

    # ---- plot loss curve ----
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Pretrain Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Pretraining Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('pretrain_loss_curve.png')
        plt.close()
        logging.info('Loss curve saved to pretrain_loss_curve.png')
    except ImportError:
        logging.warning('Matplotlib not installed. Skipping loss plot.')

    # ---- save loss history ----
    try:
        data_to_save = np.column_stack((range(1, len(epoch_losses) + 1), epoch_losses))
        np.savetxt('pretrain_loss_history.csv', data_to_save, delimiter=',',
                   header='epoch,pretrain_loss', comments='')
        logging.info('Loss history saved to pretrain_loss_history.csv')
    except Exception as e:
        logging.warning(f'Failed to save loss history: {e}')

    return epoch_losses
