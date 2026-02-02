import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, roc_auc_score, precision_recall_curve
from .utils import is_main_process, reduce_sum, distributed_concat, broadcast_value

def train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=1,
                scaler=None, world_size=1, rank=0, log_interval=5):
    """训练一个 Epoch"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros(1, device=device)
    total_correct = torch.zeros(1, device=device)
    total_samples = torch.zeros(1, device=device)

    amp_enabled = scaler is not None and scaler.is_enabled()
    autocast_device = 'cuda' if 'cuda' in device else 'cpu'

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device).float()
        target = target.to(device).long()
        batch_size = target.size(0)

        if torch.isnan(data).any():
            if is_main_process(rank):
                print(f"⚠️  输入数据在batch {batch_idx} 含NaN，跳过该batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        autocast_cm = torch.amp.autocast if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast
        autocast_kwargs = {'device_type': autocast_device, 'enabled': amp_enabled}
        if autocast_device == 'cuda':
            autocast_kwargs['dtype'] = torch.float16
            
        with autocast_cm(**autocast_kwargs):
            output_dict = model(data)
            logits = output_dict['clipwise_output']
            raw_loss = criterion(logits, target)

        if not torch.isfinite(raw_loss):
            if is_main_process(rank):
                print(f"⚠️  检测到非有限loss (batch {batch_idx})，清零梯度并跳过")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss = raw_loss / accumulation_steps

        if amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader))
        if should_step:
            if amp_enabled:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += raw_loss.detach() * batch_size
        preds = logits.argmax(dim=1)
        total_correct += preds.eq(target).sum()
        total_samples += batch_size

        if is_main_process(rank) and batch_idx % log_interval == 0:
            print(f'Batch {batch_idx}, Loss: {raw_loss.item():.6f}')

    total_loss = reduce_sum(total_loss, world_size)
    total_correct = reduce_sum(total_correct, world_size)
    total_samples = reduce_sum(total_samples, world_size)

    avg_loss = (total_loss / total_samples).item()
    accuracy = (total_correct / total_samples * 100).item()
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device, threshold=0.5, world_size=1):
    """验证模型"""
    model.eval()
    total_loss = torch.zeros(1, device=device)
    total_correct = torch.zeros(1, device=device)
    total_samples = torch.zeros(1, device=device)
    prob_chunks, target_chunks, pred_chunks = [], [], []

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device).float()
            target = target.to(device).long()

            output_dict = model(data)
            logits = output_dict['clipwise_output']
            loss = criterion(logits, target)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= threshold).long()

            batch_size = target.size(0)
            total_loss += loss.detach() * batch_size
            total_correct += preds.eq(target).sum()
            total_samples += batch_size

            prob_chunks.append(probs.detach())
            target_chunks.append(target.detach())
            pred_chunks.append(preds.detach())

    if len(prob_chunks) == 0:
        prob_tensor = torch.zeros(0, device=device)
        target_tensor = torch.zeros(0, device=device, dtype=torch.long)
        pred_tensor = torch.zeros(0, device=device, dtype=torch.long)
    else:
        prob_tensor = torch.cat(prob_chunks)
        target_tensor = torch.cat(target_chunks)
        pred_tensor = torch.cat(pred_chunks)

    total_loss = reduce_sum(total_loss, world_size)
    total_correct = reduce_sum(total_correct, world_size)
    total_samples = reduce_sum(total_samples, world_size)

    if world_size > 1:
        prob_tensor = distributed_concat(prob_tensor, world_size)
        target_tensor = distributed_concat(target_tensor, world_size)
        pred_tensor = distributed_concat(pred_tensor, world_size)

    avg_loss = (total_loss / total_samples).item()
    accuracy = (total_correct / total_samples * 100).item()

    all_probs = prob_tensor.cpu().numpy()
    all_targets = target_tensor.cpu().numpy()
    all_preds = pred_tensor.cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary', zero_division=0)
    balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100
    try:
        auc_roc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc_roc = 0.0

    return avg_loss, accuracy, precision, recall, f1, balanced_acc, auc_roc

def optimize_threshold(model, val_loader, device, world_size=1, rank=0):
    """优化分类阈值"""
    model.eval()
    probs_list = []
    targets_list = []

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            logits = model(data)['clipwise_output']
            prob = F.softmax(logits, dim=1)[:, 1]
            probs_list.append(prob.detach())
            targets_list.append(target.detach())

    if len(probs_list) == 0:
        prob_tensor = torch.zeros(0, device=device)
        target_tensor = torch.zeros(0, device=device, dtype=torch.long)
    else:
        prob_tensor = torch.cat(probs_list)
        target_tensor = torch.cat(targets_list)

    if world_size > 1:
        prob_tensor = distributed_concat(prob_tensor, world_size)
        target_tensor = distributed_concat(target_tensor, world_size)

    probs_np = prob_tensor.cpu().numpy()
    targets_np = target_tensor.cpu().numpy()

    if is_main_process(rank) and len(probs_np) > 0:
        precisions, recalls, thresholds = precision_recall_curve(targets_np, probs_np)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = float(thresholds[best_idx])
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]
        best_f1 = f1_scores[best_idx]

        print(f"\n🎯 阈值优化结果: 阈值={best_threshold:.4f}, F1={best_f1:.4f}")
    else:
        best_threshold = 0.5

    best_threshold = broadcast_value(best_threshold, device, world_size)
    return best_threshold
