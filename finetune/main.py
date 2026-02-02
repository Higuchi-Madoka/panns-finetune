import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
import numpy as np

from .utils import (setup_distributed_environment, cleanup_distributed_environment, 
                   is_main_process, create_performance_tracker, log_performance, 
                   create_training_visualization)
from .dataset import EnhancedViolenceDataset
from .models import Transfer_Cnn14_Violence
from .losses import FocalLoss
from .trainer import train_epoch, validate, optimize_threshold

def train(args):
    use_ddp, rank, world_size, local_rank = setup_distributed_environment()
    main_process = is_main_process(rank)
    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
    
    # 创建时间戳日志目录
    log_dir = os.path.join('logs', timestamp)
    if main_process:
        os.makedirs(log_dir, exist_ok=True)

    # 设备配置
    if use_ddp and torch.cuda.is_available():
        device = f'cuda:{local_rank}'
    elif args.cuda and torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available() and args.mps:
        device = 'mps'
    else:
        device = 'cpu'

    if main_process:
        print(f'🚀 启动训练 (Device: {device}, DDP: {use_ddp})')

    # 数据集
    train_dataset = EnhancedViolenceDataset(
        data_dir='dataset/train',
        sample_rate=args.sample_rate,
        clip_samples=args.sample_rate * 30,
        augment=True
    )
    test_dataset = EnhancedViolenceDataset(
        data_dir='dataset/test',
        sample_rate=args.sample_rate,
        clip_samples=args.sample_rate * 30,
        augment=False
    )

    # DataLoader
    num_workers = 0 if 'mps' in str(device) else (2 if use_ddp else 4)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=num_workers, pin_memory=('cuda' in str(device)), drop_last=use_ddp
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=test_sampler, num_workers=num_workers, pin_memory=('cuda' in str(device))
    )

    # 模型
    model = Transfer_Cnn14_Violence(
        sample_rate=args.sample_rate, window_size=args.window_size, 
        hop_size=args.hop_size, mel_bins=args.mel_bins, 
        fmin=args.fmin, fmax=args.fmax, freeze_base=args.freeze_base
    )

    # 加载预训练
    if args.pretrained_checkpoint_path:
        if main_process: print(f'Loading pretrained: {args.pretrained_checkpoint_path}')
        model.load_from_pretrain(args.pretrained_checkpoint_path)

    model.to(device)
    if use_ddp:
        if args.sync_bn: model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank] if 'cuda' in str(device) else None, find_unused_parameters=False)

    # 损失函数 (Focal Loss)
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    
    alpha_detected = (pos_count / len(train_labels)) * 0.9
    alpha_undetected = (neg_count / len(train_labels)) * 1.1
    total_w = alpha_detected + alpha_undetected
    focal_alpha = torch.FloatTensor([alpha_undetected/total_w, alpha_detected/total_w]).to(device)
    criterion = FocalLoss(alpha=focal_alpha, gamma=2.0)

    # 优化器
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 混合精度
    use_amp = args.amp and 'cuda' in str(device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler(enabled=use_amp)

    # 记录器
    performance_file = create_performance_tracker(log_dir) if main_process else None
    
    # 训练循环
    best_f1 = 0.0
    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 
               'f1': [], 'precision': [], 'recall': [], 'balanced_acc': [], 'lr': []}

    for epoch in range(args.start_epoch, args.epochs):
        if use_ddp: train_loader.sampler.set_epoch(epoch)
        
        if main_process: print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # 渐进式解冻
        original_model = model.module if use_ddp else model
        if original_model.progressive_unfreeze(epoch, args.epochs):
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                   lr=optimizer.param_groups[0]['lr'], weight_decay=args.weight_decay)
            if main_process: print("🔄 优化器已更新 (解冻新层)")

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, 
                                           args.accumulation_steps, scaler, world_size, rank)
        
        # 验证
        optimal_threshold = optimize_threshold(model, test_loader, device, world_size, rank)
        val_loss, val_acc, prec, rec, f1, bal_acc, auc = validate(model, test_loader, criterion, device, optimal_threshold, world_size)
        
        # 记录
        metrics['train_loss'].append(train_loss); metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss); metrics['val_acc'].append(val_acc)
        metrics['f1'].append(f1); metrics['precision'].append(prec); metrics['recall'].append(rec)
        metrics['balanced_acc'].append(bal_acc); metrics['lr'].append(optimizer.param_groups[0]['lr'])

        if main_process:
            print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'Test Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {f1:.4f}')
            
            if f1 > best_f1:
                best_f1 = f1
                model_path = os.path.join(log_dir, f'best_model_f1_{f1:.4f}.pth')
                torch.save(original_model.state_dict(), model_path)
                print("🏆 Best model saved!")

            if performance_file:
                log_performance(performance_file, 'ViolenceDetector', epoch+1, train_loss, train_acc, 
                              val_loss, val_acc, bal_acc, prec, rec, f1, auc, metrics['lr'][-1])
                create_training_visualization(metrics['train_loss'], metrics['train_acc'], metrics['val_loss'], 
                                            metrics['val_acc'], metrics['balanced_acc'], metrics['precision'], 
                                            metrics['recall'], metrics['f1'], metrics['lr'], log_dir)
        
        scheduler.step()

    cleanup_distributed_environment()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--window_size', type=int, default=512)
    parser.add_argument('--hop_size', type=int, default=160)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=8000)
    parser.add_argument('--pretrained_checkpoint_path', type=str, default='Cnn14_16k_mAP=0.438.pth')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--freeze_base', action='store_true', default=True)
    parser.add_argument('--no_freeze', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--mps', action='store_true', default=True)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--sync_bn', action='store_true', default=False)
    
    args = parser.parse_args()
    if args.no_freeze: args.freeze_base = False
    
    train(args)
