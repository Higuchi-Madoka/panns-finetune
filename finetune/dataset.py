import os
import random
import numpy as np
import librosa
import torch
import torch.utils.data

class SimpleAudioAugmentation:
    """音频数据增强，用于数据平衡场景"""
    def __init__(self, sample_rate=16000, target_length=160000):
        self.sample_rate = sample_rate
        self.target_length = target_length
    
    def __call__(self, waveform, label, augment_prob=0.3):
        """应用数据增强"""
        if random.random() > augment_prob:
            return waveform
        
        # 1. 音量调整 
        if random.random() < 0.3:
            volume_factor = random.uniform(0.85, 1.15)
            waveform = waveform * volume_factor
        
        # 2. 添加噪声
        if random.random() < 0.2:
            noise_factor = random.uniform(0.001, 0.005)
            noise = torch.randn_like(waveform) * noise_factor
            waveform = waveform + noise
        
        # 3. 时间移位
        if random.random() < 0.25:
            shift_amount = random.randint(-self.target_length//15, self.target_length//15)
            waveform = torch.roll(waveform, shifts=shift_amount, dims=0)
        
        # 4. 时间掩码
        if random.random() < 0.15:
            mask_length = int(len(waveform) * random.uniform(0.01, 0.05))
            if mask_length > 0 and mask_length < len(waveform):
                start = random.randint(0, len(waveform) - mask_length)
                waveform[start:start + mask_length] = 0
        
        # 5. 极性反转
        if random.random() < 0.1:
            waveform = -waveform
            
        return waveform

class KnockDataset(torch.utils.data.Dataset):
    """敲击声数据集（弃用）"""
    def __init__(self, data_dir, sample_rate=16000, clip_samples=None):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.clip_samples = clip_samples or sample_rate * 10
        
        self.audio_files = []
        self.labels = []
        
        self._scan_folder(os.path.join(data_dir, 'detected'), 1)
        self._scan_folder(os.path.join(data_dir, 'undetected'), 0)
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {data_dir}")
            
    def _scan_folder(self, folder, label):
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    self.audio_files.append(os.path.join(folder, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        if len(waveform) > self.clip_samples:
            start = (len(waveform) - self.clip_samples) // 2
            waveform = waveform[start:start + self.clip_samples]
        elif len(waveform) < self.clip_samples:
            waveform = np.pad(waveform, (0, self.clip_samples - len(waveform)), 'constant')
        
        waveform = waveform.astype(np.float32)
        return waveform, label

class EnhancedViolenceDataset(torch.utils.data.Dataset):
    """暴力事件数据集"""
    def __init__(self, data_dir, sample_rate=16000, clip_samples=None, 
                 augment=False, oversample_minority=False):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.clip_samples = clip_samples or sample_rate * 30
        self.augment = augment
        self.oversample_minority = oversample_minority
        
        self.audio_augment = SimpleAudioAugmentation(sample_rate, self.clip_samples) if augment else None
        
        self.audio_files = []
        self.labels = []
        
        self._scan_and_add(os.path.join(data_dir, 'positive'), 1)
        self._scan_and_add(os.path.join(data_dir, 'detected'), 1)
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {data_dir}")
            
        print(f"Dataset loaded from {data_dir}: {len(self.audio_files)} samples")
        pos_count = sum(self.labels)
        print(f"   Positive: {pos_count}, Negative: {len(self.labels) - pos_count}")

    def _scan_and_add(self, folder, label):
        if not os.path.exists(folder):
            return
        files = []
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                files.append(os.path.join(folder, filename))
        
        for f in files:
            self.audio_files.append(f)
            self.labels.append(label)

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        try:
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            waveform = np.zeros(self.clip_samples, dtype=np.float32)

        # 长度标准化
        if len(waveform) > self.clip_samples:
            if self.augment:
                start = random.randint(0, len(waveform) - self.clip_samples)
            else:
                start = (len(waveform) - self.clip_samples) // 2
            waveform = waveform[start:start + self.clip_samples]
        elif len(waveform) < self.clip_samples:
            waveform = np.pad(waveform, (0, self.clip_samples - len(waveform)), 'constant')
        
        waveform = torch.from_numpy(waveform.astype(np.float32))
        
        if self.augment:
            waveform = self.audio_augment(waveform, label, augment_prob=0.8)
            
        return waveform, label
