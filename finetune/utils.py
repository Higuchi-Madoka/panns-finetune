import os
import sys
import torch
import torch.distributed as dist
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def setup_distributed_environment():
    """初始化分布式环境 (DDP)"""
    if not dist.is_available() or 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, init_method='env://')
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank

def cleanup_distributed_environment():
    """清理分布式环境"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    return rank == 0

def reduce_sum(tensor, world_size):
    """跨卡求和"""
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def distributed_concat(tensor, world_size):
    """跨卡拼接 Tensor"""
    if world_size == 1:
        return tensor
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)

def broadcast_value(value, device, world_size, src=0):
    """广播单个值"""
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    if world_size > 1:
        dist.broadcast(tensor, src=src)
    return tensor.item()

def do_mixup(x, mixup_lambda):
    """Mixup 数据增强"""
    out = x[0::2].transpose(0, -1) * mixup_lambda[0::2] + \
          x[1::2].transpose(0, -1) * mixup_lambda[1::2]
    return out.transpose(0, -1)

class EarlyStopping:
    """早停机制（未启用）"""
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

def create_performance_tracker(log_dir='logs'):
    """性能跟踪CSV 文件"""
    os.makedirs(log_dir, exist_ok=True)
    performance_file = os.path.join(log_dir, 'performance.csv')
    
    if not os.path.exists(performance_file):
        columns = [
            'timestamp', 'model_name', 'epoch', 'train_loss', 'train_acc', 
            'val_loss', 'val_acc', 'balanced_acc', 'precision', 'recall', 
            'f1_score', 'auc_roc', 'learning_rate', 'dataset_type', 'augmentation'
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(performance_file, index=False)
    
    return performance_file

def log_performance(performance_file, model_name, epoch, train_loss, train_acc, 
                   val_loss, val_acc, balanced_acc, precision, recall, f1, auc_roc, lr, 
                   dataset_type='enhanced', augmentation=True):
    """记录性能指标"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = {
        'timestamp': timestamp, 'model_name': model_name, 'epoch': epoch,
        'train_loss': train_loss, 'train_acc': train_acc,
        'val_loss': val_loss, 'val_acc': val_acc,
        'balanced_acc': balanced_acc, 'precision': precision,
        'recall': recall, 'f1_score': f1, 'auc_roc': auc_roc,
        'learning_rate': lr, 'dataset_type': dataset_type, 'augmentation': augmentation
    }
    df = pd.read_csv(performance_file)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(performance_file, index=False)

def create_training_visualization(train_losses, train_accs, val_losses, val_accs, 
                                 balanced_accs, precisions, recalls, f1_scores, 
                                 learning_rates, log_dir, has_validation=True):
    """绘制训练曲线"""
    plt.figure(figsize=(16, 12))
    epochs = range(1, len(train_losses) + 1)
    
    # 1. Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Test Loss', linewidth=2)
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    
    # 2. Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Acc', linewidth=2)
    plt.plot(epochs, val_accs, 'r-', label='Test Acc', linewidth=2)
    plt.plot(epochs, balanced_accs, 'g-', label='Test Balanced Acc', linewidth=2)
    plt.title('Accuracy Metrics')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True, alpha=0.3)
    
    # 3. F1/Precision/Recall
    plt.subplot(2, 2, 3)
    plt.plot(epochs, f1_scores, 'r-', label='F1-Score', linewidth=2)
    plt.plot(epochs, precisions, 'g-', label='Precision', linewidth=2)
    plt.plot(epochs, recalls, 'b-', label='Recall', linewidth=2)
    plt.title('Classification Metrics')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.grid(True, alpha=0.3)
    
    # 4. Learning Rate
    plt.subplot(2, 2, 4)
    plt.plot(epochs, learning_rates, 'purple', linewidth=2)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch'); plt.ylabel('LR'); plt.yscale('log'); plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = os.path.join(log_dir, 'training_curves.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_filename
