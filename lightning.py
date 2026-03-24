import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from metric import MetricMonitor

class KGReasoningModule(pl.LightningModule):
    """
    PyTorch Lightning 包装器
    """
    def __init__(self, model, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.criterion = nn.MSELoss()
        
        self.train_monitor = MetricMonitor()
        self.val_monitor = MetricMonitor()
        self.test_monitor = MetricMonitor() 

    def forward(self, h, r, t, lre_data, sfe_data):
        return self.model(h, r, t, lre_data, sfe_data)

    def training_step(self, batch, batch_idx):
        h, r, t, y, lre_data, sfe_data = batch
        
        preds = self(h, r, t, lre_data, sfe_data)
        
        task_loss = self.criterion(preds, y)
        
        self.log('train_task_loss', task_loss, prog_bar=True)
        self.log('train_loss', task_loss, prog_bar=True, on_step=True, on_epoch=True)
        
        self.train_monitor.update(preds, y)
        return task_loss

    def on_train_epoch_end(self):
        metrics = self.train_monitor.get_metrics()
        print(f"\n[Train] Epoch {self.current_epoch} | MSE: {metrics['MSE']:.4f} | MAE: {metrics['MAE']:.4f}")
        self.train_monitor.reset()

    def validation_step(self, batch, batch_idx):
        h, r, t, y, lre_data, sfe_data = batch
        
        preds = self(h, r, t, lre_data, sfe_data)

        loss = self.criterion(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.val_monitor.update(preds, y)
        return loss

    def on_validation_epoch_end(self):
        metrics = self.val_monitor.get_metrics()
        print(f"[Valid] Epoch {self.current_epoch} | MSE: {metrics['MSE']:.4f} | MAE: {metrics['MAE']:.4f}")
        self.log('val_mse', metrics['MSE'])
        self.log('val_mae', metrics['MAE'])
        self.val_monitor.reset()

    def test_step(self, batch, batch_idx):
        h, r, t, y, lre_data, sfe_data = batch
        
        preds = self(h, r, t, lre_data, sfe_data)
        
        loss = self.criterion(preds, y)
        
        self.log('test_loss', loss, prog_bar=True)
        self.test_monitor.update(preds, y)
        return loss

    def on_test_epoch_end(self):
        metrics = self.test_monitor.get_metrics()
        print(f"\n[Test Result] FINAL | MSE: {metrics['MSE']:.4f} | MAE: {metrics['MAE']:.4f}")
        self.log('test_mse', metrics['MSE'])
        self.log('test_mae', metrics['MAE'])
        self.test_monitor.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
