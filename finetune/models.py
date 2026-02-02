import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

def init_layer(layer):
    """初始化线性层或卷积层"""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """初始化BatchNorm层"""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def do_mixup(x, mixup_lambda):
    """Mixup 数据增强"""
    out = x[0::2].transpose(0, -1) * mixup_lambda[0::2] + \
          x[1::2].transpose(0, -1) * mixup_lambda[1::2]
    return out.transpose(0, -1)

class ConvBlock(nn.Module):
    """卷积块"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        return x

class Cnn14_16k(nn.Module):
    """Cnn14 16kHz模型"""
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super(Cnn14_16k, self).__init__()
        assert sample_rate == 16000
        assert window_size == 512
        assert hop_size == 160
        assert mel_bins == 64
        assert fmin == 50
        assert fmax == 8000

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        return output_dict

class Transfer_Cnn14_Violence(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, freeze_base=True):
        super(Transfer_Cnn14_Violence, self).__init__()
        audioset_classes_num = 527
        classes_num = 2
        
        self.base = Cnn14_16k(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)
        self.dropout = nn.Dropout(0.5)

        self.freeze_base = freeze_base
        self.layers_to_unfreeze = self._get_unfreeze_schedule()
        self.current_unfrozen_level = 0
        
        if freeze_base:
            self._freeze_all_base_layers()

        self.init_weights()

    def _get_unfreeze_schedule(self):
        return [
            [],
            ['base.fc1', 'base.fc_audioset'],
            ['base.fc1', 'base.fc_audioset', 'base.conv_block6'],
            ['base.fc1', 'base.fc_audioset', 'base.conv_block6', 'base.conv_block5'],
            ['base.fc1', 'base.fc_audioset', 'base.conv_block6', 'base.conv_block5', 'base.conv_block4'],
            ['base.fc1', 'base.fc_audioset', 'base.conv_block6', 'base.conv_block5', 'base.conv_block4', 'base.conv_block3'],
            ['base.fc1', 'base.fc_audioset', 'base.conv_block6', 'base.conv_block5', 'base.conv_block4', 'base.conv_block3', 'base.conv_block2'],
            ['base.fc1', 'base.fc_audioset', 'base.conv_block6', 'base.conv_block5', 'base.conv_block4', 'base.conv_block3', 'base.conv_block2', 'base.conv_block1', 'base.bn0']
        ]
    
    def _freeze_all_base_layers(self):
        for param in self.base.parameters():
            param.requires_grad = False
    
    def progressive_unfreeze(self, epoch, total_epochs):
        if not self.freeze_base:
            return False
        
        if total_epochs <= 50:
            stage_length = max(total_epochs // 4, 5)
            target_level = min(epoch // stage_length, len(self.layers_to_unfreeze) - 1)
        elif total_epochs <= 100:
            if epoch <= 15: target_level = 0
            elif epoch <= 30: target_level = 1
            elif epoch <= 50: target_level = 2
            elif epoch <= 75: target_level = 3
            else: target_level = min(4 + (epoch - 75) // 10, len(self.layers_to_unfreeze) - 1)
        else:
            if epoch <= 20: target_level = 0
            elif epoch <= 40: target_level = 1
            elif epoch <= 60: target_level = 2
            elif epoch <= 100: target_level = 3
            elif epoch <= 150: target_level = 4
            elif epoch <= 200: target_level = 5
            elif epoch <= 230: target_level = 6
            else: target_level = 7
        
        if target_level > self.current_unfrozen_level:
            self._unfreeze_to_level(target_level)
            self.current_unfrozen_level = target_level
            return True
        return False
    
    def _unfreeze_to_level(self, level):
        if level >= len(self.layers_to_unfreeze):
            level = len(self.layers_to_unfreeze) - 1
        self._freeze_all_base_layers()
        layers_to_unfreeze = self.layers_to_unfreeze[level]
        for layer_name in layers_to_unfreeze:
            try:
                module = eval(f'self.{layer_name}')
                for param in module.parameters():
                    param.requires_grad = True
            except:
                pass
    
    def get_trainable_params_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'trainable_ratio': trainable_params / total_params * 100
        }

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        try:
            checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu', weights_only=True)
        except Exception:
            print("Warning: Loading checkpoint with weights_only=False")
            checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu', weights_only=False)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']
        embedding = self.dropout(embedding)
        clipwise_output = self.fc_transfer(embedding)
        output_dict['clipwise_output'] = clipwise_output
        return output_dict
