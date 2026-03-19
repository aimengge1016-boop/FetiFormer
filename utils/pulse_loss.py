"""
脉冲敏感损失函数 (Pulse-Sensitive Loss)

结合了 MSE (全局拟合) 和 梯度加权惩罚 (局部突变捕捉)

主要特点：
1. 动态检测真实值中的突变点（通过一阶差分）
2. 对脉冲点进行加权补偿
3. 防止模型"平滑"掉高频脉冲信号
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PulseSensitiveLoss(nn.Module):
    """
    脉冲敏感损失函数
    
    创新点：
    - 使用一阶差分识别突变点
    - 动态阈值计算（基于 batch 统计量）
    - 脉冲位置加权惩罚
    
    参数:
        pulse_weight (float): 脉冲点的额外权重，默认 2.0
        threshold_sigma (float): 判定为脉冲的方差倍数，默认 2.0
        min_pulse_weight (float): 最小脉冲权重，默认 1.0
        use_relative_weight (bool): 是否使用相对权重（考虑脉冲幅度）
    """
    
    def __init__(self, pulse_weight=2.0, threshold_sigma=2.0, 
                 min_pulse_weight=1.0, use_relative_weight=True):
        super(PulseSensitiveLoss, self).__init__()
        
        self.pulse_weight = pulse_weight
        self.threshold_sigma = threshold_sigma
        self.min_pulse_weight = min_pulse_weight
        self.use_relative_weight = use_relative_weight
    
    def forward(self, pred, target):
        """
        计算脉冲敏感损失
        
        Args:
            pred: 预测值 [B, L, N] 或 [B, L]
            target: 目标值 [B, L, N] 或 [B, L]
        
        Returns:
            loss: 标量损失值
        """
        # 确保形状一致
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        
        # 1. 基础 MSE 损失
        base_loss = F.mse_loss(pred, target, reduction='mean')
        
        # 处理不同维度情况
        if pred.dim() == 3:
            # [B, L, N] - 多变量情况
            loss = self._compute_3d_loss(pred, target, base_loss)
        elif pred.dim() == 2:
            # [B, L] - 单变量情况
            loss = self._compute_2d_loss(pred, target, base_loss)
        else:
            # 回退到普通 MSE
            loss = base_loss
        
        return loss
    
    def _compute_3d_loss(self, pred, target, base_loss):
        """
        处理 3D 张量 [B, L, N]
        """
        B, L, N = target.shape
        
        # 2. 计算目标值的变化率（一阶差分），识别突变点
        # diff: [B, L-1, N]
        diff = torch.abs(target[:, 1:, :] - target[:, :-1, :])
        
        # 3. 动态计算脉冲阈值
        # 计算 batch 内的均值和标准差
        mu = torch.mean(diff)
        std = torch.std(diff)
        threshold = mu + self.threshold_sigma * std
        
        # 4. 识别脉冲掩码
        pulse_mask = (diff > threshold).float()  # [B, L-1, N]
        
        # 统计脉冲点数量（用于调试）
        pulse_ratio = torch.mean(pulse_mask).item()
        
        # 5. 计算预测值的变化率
        pred_diff = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])
        
        # 6. 脉冲损失：如果真实值波动剧烈而预测值波动小，则加大惩罚
        # 核心思想：确保预测值在脉冲位置的变化幅度与真实值相近
        
        if self.use_relative_weight:
            # 相对权重：脉冲幅度越大，惩罚越重
            relative_weight = torch.clamp(diff / (torch.mean(torch.abs(target)) + 1e-8), 
                                          min=1.0, max=10.0)
            pulse_loss = torch.mean(pulse_mask * relative_weight * torch.square(pred_diff - diff))
        else:
            pulse_loss = torch.mean(pulse_mask * torch.square(pred_diff - diff))
        
        # 7. 额外惩罚：确保脉冲位置的值本身也准确
        # 如果预测值在脉冲点的绝对值偏差过大，也应该受到惩罚
        pulse_position_loss = torch.mean(pulse_mask * torch.abs(pred[:, 1:, :] - target[:, 1:, :]))
        
        # 组合损失
        total_loss = base_loss + self.pulse_weight * pulse_loss + 0.5 * pulse_position_loss
        
        return total_loss
    
    def _compute_2d_loss(self, pred, target, base_loss):
        """
        处理 2D 张量 [B, L]
        """
        # 2. 计算一阶差分
        diff = torch.abs(target[:, 1:] - target[:, :-1])  # [B, L-1]
        
        # 3. 动态计算脉冲阈值
        mu = torch.mean(diff)
        std = torch.std(diff)
        threshold = mu + self.threshold_sigma * std
        
        # 4. 识别脉冲掩码
        pulse_mask = (diff > threshold).float()  # [B, L-1]
        
        # 5. 计算预测值的变化率
        pred_diff = torch.abs(pred[:, 1:] - pred[:, :-1])
        
        # 6. 脉冲损失
        if self.use_relative_weight:
            relative_weight = torch.clamp(diff / (torch.mean(torch.abs(target)) + 1e-8),
                                          min=1.0, max=10.0)
            pulse_loss = torch.mean(pulse_mask * relative_weight * torch.square(pred_diff - diff))
        else:
            pulse_loss = torch.mean(pulse_mask * torch.square(pred_diff - diff))
        
        # 7. 脉冲位置损失
        pulse_position_loss = torch.mean(pulse_mask * torch.abs(pred[:, 1:] - target[:, 1:]))
        
        # 组合损失
        total_loss = base_loss + self.pulse_weight * pulse_loss + 0.5 * pulse_position_loss
        
        return total_loss


class AdaptivePulseLoss(nn.Module):
    """
    自适应脉冲损失 - 根据数据特性自动调整参数
    
    增强特性：
    1. 自动检测数据中的脉冲密度
    2. 动态调整脉冲权重
    3. 多尺度脉冲检测
    """
    
    def __init__(self, base_pulse_weight=2.0, scale_factors=[1, 2, 4]):
        super(AdaptivePulseLoss, self).__init__()
        
        self.base_pulse_weight = base_pulse_weight
        self.scale_factors = scale_factors
        
        # 可学习的脉冲权重
        self.pulse_weight = nn.Parameter(torch.tensor(base_pulse_weight))
    
    def forward(self, pred, target):
        """
        计算自适应脉冲敏感损失
        """
        B, L, N = target.shape
        
        # 1. 基础 MSE
        base_loss = F.mse_loss(pred, target, reduction='mean')
        
        total_pulse_loss = 0.0
        total_weight = 0.0
        
        # 多尺度脉冲检测
        for scale in self.scale_factors:
            if scale == 1:
                # 原始尺度的差分
                diff = torch.abs(target[:, 1:, :] - target[:, :-1, :])
                pred_diff = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])
            else:
                # 多尺度差分
                # 使用池化来捕捉不同时间尺度的变化
                diff = self._multiscale_diff(target, scale)
                pred_diff = self._multiscale_diff(pred, scale)
            
            # 动态阈值
            mu = torch.mean(diff)
            std = torch.std(diff)
            threshold = mu + 2.0 * std
            
            # 脉冲掩码
            pulse_mask = (diff > threshold).float()
            
            # 尺度特定的脉冲损失
            scale_weight = 1.0 / scale  # 较大尺度权重较低
            scale_loss = torch.mean(pulse_mask * torch.square(pred_diff - diff))
            
            total_pulse_loss += scale_weight * scale_loss
            total_weight += scale_weight
        
        # 平均多尺度损失
        avg_pulse_loss = total_pulse_loss / (total_weight + 1e-8)
        
        # 最终损失
        total_loss = base_loss + torch.abs(self.pulse_weight) * avg_pulse_loss
        
        return total_loss
    
    def _multiscale_diff(self, x, scale):
        """
        计算多尺度差分
        """
        if scale == 1:
            return torch.abs(x[:, 1:, :] - x[:, :-1, :])
        
        # 对于更大的尺度，使用滑动窗口差分
        diff = torch.abs(x[:, scale:, :] - x[:, :-scale, :])
        return diff


class PulseLossFactory:
    """
    损失函数工厂类
    
    方便创建不同类型的脉冲损失函数
    """
    
    @staticmethod
    def create_loss(loss_type='pulse', **kwargs):
        """
        创建损失函数
        
        Args:
            loss_type: 损失类型
                - 'pulse': 标准脉冲敏感损失
                - 'adaptive': 自适应脉冲损失
                - 'combined': 组合损失 (MSE + Pulse)
            **kwargs: 其他参数
        """
        if loss_type == 'pulse':
            return PulseSensitiveLoss(
                pulse_weight=kwargs.get('pulse_weight', 2.0),
                threshold_sigma=kwargs.get('threshold_sigma', 2.0),
                min_pulse_weight=kwargs.get('min_pulse_weight', 1.0),
                use_relative_weight=kwargs.get('use_relative_weight', True)
            )
        elif loss_type == 'adaptive':
            return AdaptivePulseLoss(
                base_pulse_weight=kwargs.get('base_pulse_weight', 2.0),
                scale_factors=kwargs.get('scale_factors', [1, 2, 4])
            )
        elif loss_type == 'combined':
            # 返回组合损失 (可以在外部与 MSE 结合)
            return PulseSensitiveLoss(**kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


# 便捷函数
def get_pulse_loss(config):
    """
    根据配置获取脉冲损失函数
    
    Args:
        config: argparse 配置对象
    
    Returns:
        损失函数实例
    """
    if hasattr(config, 'use_pulse_loss') and config.use_pulse_loss:
        return PulseLossFactory.create_loss(
            loss_type=getattr(config, 'pulse_loss_type', 'pulse'),
            pulse_weight=getattr(config, 'pulse_weight', 2.0),
            threshold_sigma=getattr(config, 'pulse_threshold_sigma', 2.0)
        )
    else:
        return nn.MSELoss()

