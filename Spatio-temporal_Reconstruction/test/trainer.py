import torch
import torch.nn as nn
import numpy as np
import logging
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

logger = logging.getLogger(__name__)

def compute_metrics(y_true, y_pred):
    """返回 R2, RMSE"""
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = float('nan')
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse

class Trainer:
    """通用 Trainer，支持训练/验证循环，并保存最优模型（以 R2 为准或以 val loss）"""
    def __init__(self, model, optimizer, device='cuda', criterion=None, scheduler=None, grad_clip=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.scheduler = scheduler
        self.grad_clip = grad_clip

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        cnt = 0
        for batch in tqdm(dataloader, desc="train"):
            # 预期 batch 包含: num (B,T,F), time/loc 已并入 num, cat dict, y (B)
            x_num = batch['num'].to(self.device)       # B x T x F
            cat = {k: v.to(self.device) for k, v in batch.get('cat', {}).items()}
            y = batch['y'].to(self.device)             # B

            preds = self.model(x_num, cat_dict=cat)
            loss = self.criterion(preds, y)
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item() * y.size(0)
            cnt += y.size(0)
        avg_loss = total_loss / max(1, cnt)
        return avg_loss

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        all_y = []
        all_p = []
        total_loss = 0.0
        cnt = 0
        for batch in tqdm(dataloader, desc="val"):
            x_num = batch['num'].to(self.device)
            cat = {k: v.to(self.device) for k, v in batch.get('cat', {}).items()}
            y = batch['y'].to(self.device)

            preds = self.model(x_num, cat_dict=cat)
            loss = self.criterion(preds, y)
            total_loss += loss.item() * y.size(0)
            cnt += y.size(0)

            all_y.append(y.detach().cpu().numpy())
            all_p.append(preds.detach().cpu().numpy())

        if cnt == 0:
            return {'loss': float('nan'), 'r2': float('nan'), 'rmse': float('nan')}

        y_all = np.concatenate(all_y)
        p_all = np.concatenate(all_p)
        r2, rmse = compute_metrics(y_all, p_all)
        avg_loss = total_loss / cnt
        return {'loss': avg_loss, 'r2': r2, 'rmse': rmse}
