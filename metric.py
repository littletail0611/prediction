import torch

class MetricMonitor:
    """
    指标监控器
    用于在训练/验证过程中累积每个 Batch 的结果，并计算全局 MSE 和 MAE。
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """重置累积器，通常在一个 Epoch 开始时调用"""
        self.mse_sum = 0.0
        self.mae_sum = 0.0
        self.n_samples = 0

    @torch.no_grad()
    def update(self, preds, labels):
        """
        更新当前 Batch 的指标
        :param preds: 模型预测值 (Tensor), shape [Batch] or [Batch, 1]
        :param labels: 真实标签 (Tensor), shape [Batch] or [Batch, 1]
        """
        # 1. 确保数据在 CPU 上并展平为一维向量
        p = preds.detach().view(-1).cpu()
        t = labels.detach().view(-1).cpu()
        
        # 2. 计算当前 Batch 的误差和
        # MSE: (y - y_hat)^2
        batch_mse_sum = torch.sum((p - t) ** 2).item()
        
        # MAE: |y - y_hat|
        batch_mae_sum = torch.sum(torch.abs(p - t)).item()
        
        # 3. 累积
        self.mse_sum += batch_mse_sum
        self.mae_sum += batch_mae_sum
        self.n_samples += p.numel()

    def get_metrics(self):
        """
        计算并返回当前累积的平均指标
        :return: dict {'MSE': float, 'MAE': float}
        """
        if self.n_samples == 0:
            return {"MSE": 0.0, "MAE": 0.0}

        mse = self.mse_sum / self.n_samples
        mae = self.mae_sum / self.n_samples
        
        return {
            "MSE": mse,
            "MAE": mae
        }

    def print_metrics(self, prefix="Valid"):
        """
        辅助函数：格式化打印
        """
        metrics = self.get_metrics()
        print(f"[{prefix}] MSE: {metrics['MSE']:.6f} | MAE: {metrics['MAE']:.6f}")
        return metrics

# 简单的函数式接口（如果不想用类的话）
def calculate_batch_metrics(preds, labels):
    p = preds.detach().view(-1)
    t = labels.detach().view(-1)
    
    mse = torch.mean((p - t) ** 2).item()
    mae = torch.mean(torch.abs(p - t)).item()
    
    return mse, mae