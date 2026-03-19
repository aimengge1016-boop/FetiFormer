"""
注意力权重可视化脚本
====================
生成 Query-Key 相关性热力图，展示模型在长序列预测时的注意力分布。

使用方法:
    python visualize_attention.py --model_path checkpoints/xxx/checkpoint.pth --seq_len 96 --pred_len 96

输出:
    - attention_heatmap.png: 注意力权重热力图
    - attention_analysis.png: 多维度注意力分析图
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def to_numpy(tensor):
    """安全地将张量转换为 numpy 数组（处理 CUDA 张量）"""
    if hasattr(tensor, 'cpu'):
        tensor = tensor.cpu()
    if hasattr(tensor, 'numpy'):
        return tensor.numpy()
    return np.array(tensor)
import argparse
import os
import sys
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 添加项目路径
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)


def set_seed(seed=2023):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model_and_data(args):
    """加载模型和测试数据"""
    from data_provider.data_factory import data_provider
    
    # 加载检查点来获取真实的模型参数
    print(f"Loading checkpoint to extract parameters: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # 从检查点推断所有真实的模型参数
    actual_d_model = 512
    actual_pred_len = 720
    actual_seq_len = 96
    actual_n_heads = 8
    actual_e_layers = 2
    actual_d_ff = 512
    
    # projector.2.weight 形状为 [pred_len, d_model]
    if 'projector.2.weight' in checkpoint:
        p2_shape = checkpoint['projector.2.weight'].shape
        actual_pred_len = int(p2_shape[0])
        actual_d_model = int(p2_shape[1])
        print(f"  From projector.2.weight: pred_len={actual_pred_len}, d_model={actual_d_model}")
    
    # enc_embedding.value_embedding.weight 形状为 [d_model, seq_len]
    if 'enc_embedding.value_embedding.weight' in checkpoint:
        emb_shape = checkpoint['enc_embedding.value_embedding.weight'].shape
        actual_seq_len = int(emb_shape[1])
        print(f"  From embedding: seq_len={actual_seq_len}")
    
    # n_heads 从 inner_attention.temp 推断
    if 'encoder.attn_layers.0.attention.inner_attention.temp' in checkpoint:
        actual_n_heads = int(checkpoint['encoder.attn_layers.0.attention.inner_attention.temp'].shape[0])
        print(f"  From temp: n_heads={actual_n_heads}")
    
    # e_layers 从层数推断
    max_layer = 0
    for key in checkpoint.keys():
        if 'attn_layers.' in key:
            layer_match = re.search(r'attn_layers\.(\d+)\.', key)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                max_layer = max(max_layer, layer_idx)
    actual_e_layers = max_layer + 1
    print(f"  From layers: e_layers={actual_e_layers}")
    
    # 检查是否存在 TSSA 相关的参数
    has_tssa = any('inner_attention' in k for k in checkpoint.keys())
    has_standard = any('query_projection' in k for k in checkpoint.keys())
    print(f"  Checkpoint type: {'TSSA' if has_tssa else 'Standard Attention'}")
    
    # 更新 args
    args.d_model = actual_d_model
    args.pred_len = actual_pred_len
    args.seq_len = actual_seq_len
    args.n_heads = actual_n_heads
    args.e_layers = actual_e_layers
    args.d_ff = actual_d_ff
    
    # 添加模型需要的其他属性
    args.output_attention = True
    args.use_norm = True
    args.use_time2vec = True
    args.dropout = 0.1
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.freq = 'h'
    args.num_workers = 0
    args.batch_size = 8
    args.enc_in = 7  # ETTh1 has 7 features
    args.dec_in = 7
    args.c_out = 7
    args.class_strategy = 'average'  # 添加缺失参数
    args.use_aefin = False
    args.freq_topk = 1
    args.inverse = False
    
    print(f"\n  Final model parameters:")
    print(f"    seq_len: {args.seq_len}")
    print(f"    pred_len: {args.pred_len}")
    print(f"    d_model: {args.d_model}")
    print(f"    n_heads: {args.n_heads}")
    print(f"    e_layers: {args.e_layers}")
    print(f"    d_ff: {args.d_ff}")
    
    # 根据检查点类型选择正确的模型
    if has_tssa:
        print("  Using TSSA model...")
        # 确保 args 包含所需属性
        args.output_attention = True
        args.use_norm = getattr(args, 'use_norm', True)
        args.use_time2vec = getattr(args, 'use_time2vec', True)
        from model.iTransformer import Model
        model = Model(args).float()
        model.load_state_dict(checkpoint, strict=False)
    elif has_standard:
        print("  Using Standard Attention model...")
        # 导入原始版本
        import importlib.util
        spec = importlib.util.spec_from_file_location("iTransformer_standard", 
                                                      os.path.join(args.project_root, "model", "iTransformer原始.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 修改原始版本的模型以匹配检查点中的 projector 结构
        class StandardModelWithProjector(module.Model):
            def __init__(self, configs):
                super().__init__(configs)
                # 替换 projector 为 3 层结构
                self.projector = nn.Sequential(
                    nn.Linear(configs.d_model, configs.d_model),
                    nn.GELU(),
                    nn.Linear(configs.d_model, configs.pred_len)
                )
        
        # 创建标准版本的模型配置
        args_standard = argparse.Namespace(
            seq_len=actual_seq_len if 'actual_seq_len' in dir() else args.seq_len,
            pred_len=actual_pred_len,
            label_len=args.label_len,
            d_model=actual_d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            embed=args.embed,
            activation=args.activation,
            enc_in=args.enc_in,
            dec_in=args.dec_in,
            c_out=args.c_out,
            freq=args.freq,
            output_attention=True,
            use_norm=args.use_norm,
            class_strategy='pred',
        )
        model = StandardModelWithProjector(args_standard)
        model.load_state_dict(checkpoint, strict=False)
    else:
        raise ValueError("Unknown checkpoint format")
    
    model.eval()
    model.to(args.device)
    
    # 获取测试数据
    _, test_loader = data_provider(args, flag='test')
    
    return model, test_loader, args


def extract_attention_weights(model, test_loader, device, args, max_batches=5):
    """从模型中提取注意力权重"""
    model.eval()
    all_attentions = []
    
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
            
            batch_x = batch_x.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device) if batch_x_mark is not None else None
            
            # 设置 output_attention=True 以获取注意力权重
            # 需要临时修改模型配置
            original_output_attention = model.output_attention
            model.output_attention = True
            
            # 前向传播
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            try:
                outputs, attentions = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            except Exception as e:
                print(f"Error in forward pass: {e}")
                continue
            
            model.output_attention = original_output_attention
            
            # 收集注意力权重
            # attentions: list of [B, N, H, N] for each layer
            all_attentions.append({
                'batch_idx': batch_idx,
                'attentions': attentions,
                'input': to_numpy(batch_x)
            })
            
            print(f"Batch {batch_idx}: Extracted attention weights from {len(attentions)} layers")
    
    return all_attentions


def visualize_attention_heatmap(attention_matrix, seq_len, save_path, title="Query-Key Attention Heatmap"):
    """
    绘制单个注意力权重热力图

    Args:
        attention_matrix: [N, N] 或 [H, N, N] 或 [B, H, N, N]
        seq_len: 序列长度
        save_path: 保存路径
        title: 标题
    """

    def _draw_heatmap(data, ax, cmap_name='YlOrRd'):
        """绘制热力图"""
        if data.ndim == 2:
            im = ax.imshow(data, cmap=cmap_name, aspect='auto',
                          interpolation='nearest')
        elif data.ndim == 3:
            # 多头注意力：取平均
            data_mean = data.mean(axis=0)
            im = ax.imshow(data_mean, cmap=cmap_name, aspect='auto',
                          interpolation='nearest')
        return im

    def _save_subplot_as_pdf(data, save_path, title, xlabel, ylabel, is_heatmap=True):
        """将单个子图保存为 PDF 文件"""
        fig, ax = plt.subplots(figsize=(8, 6))

        if is_heatmap:
            im = ax.imshow(data, cmap='YlOrRd', aspect='auto',
                          interpolation='nearest', vmin=0, vmax=1)
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Attention Weight', fontsize=11)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
        else:
            ax.plot(data['x'], data['mean'], 'b-', linewidth=1.5)
            if 'std' in data:
                ax.fill_between(data['x'], data['mean'] - data['std'],
                               data['mean'] + data['std'], alpha=0.3)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

        # 设置刻度
        n = data.shape[0] if is_heatmap else len(data['x'])
        if is_heatmap:
            tick_step = max(1, n // 8)
            tick_positions = np.arange(0, n, tick_step)
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels([str(i + 1) for i in tick_positions])
            ax.set_yticklabels([str(i + 1) for i in tick_positions])

        plt.tight_layout()
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"  Saved PDF: {save_path}")

    # 获取保存目录
    output_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 原始注意力权重
    ax1 = axes[0]
    if attention_matrix.ndim == 4:
        # [B, H, N, N] -> 取第一个batch，第一个head
        data1 = attention_matrix[0, 0, :seq_len, :seq_len]
    elif attention_matrix.ndim == 3:
        data1 = attention_matrix[0, :seq_len, :seq_len]
    else:
        data1 = attention_matrix[:seq_len, :seq_len]

    im1 = _draw_heatmap(data1, ax1)
    ax1.set_title(f'Head 1 Attention\n(First Batch)', fontsize=12)
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 保存子图1为 PDF
    _save_subplot_as_pdf(
        to_numpy(data1),
        os.path.join(output_dir, f'{base_name}_head1.pdf'),
        'Head 1 Attention Weight Matrix',
        'Key Position',
        'Query Position'
    )

    # 2. 多头平均注意力
    ax2 = axes[1]
    if attention_matrix.ndim == 4:
        # [B, N, H, N] -> 对所有 heads 取平均得到 [N, N]
        data2 = attention_matrix[0, :, :, :].mean(axis=1)
        data2 = data2[:seq_len, :seq_len]
        n_heads = attention_matrix.shape[2]
    elif attention_matrix.ndim == 3:
        # [H, N, N] -> 对所有 heads 取平均
        data2 = attention_matrix.mean(axis=0)[:seq_len, :seq_len]
        n_heads = attention_matrix.shape[0]
    else:
        data2 = attention_matrix[:seq_len, :seq_len]
        n_heads = 1

    im2 = _draw_heatmap(data2, ax2)
    ax2.set_title(f'Average Multi-Head Attention\n(All {n_heads} Heads)', fontsize=12)
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # 保存子图2为 PDF
    _save_subplot_as_pdf(
        to_numpy(data2),
        os.path.join(output_dir, f'{base_name}_multihead_avg.pdf'),
        f'Average Multi-Head Attention ({n_heads} Heads)',
        'Key Position',
        'Query Position'
    )

    # 3. 注意力权重沿对角线的统计
    ax3 = axes[2]
    if attention_matrix.ndim == 4:
        # [B, N, H, N] -> [N, H, N] (取第一个batch)
        # 然后对所有 heads 取平均得到 [N, N]
        data3 = attention_matrix[0, :, :, :].mean(axis=1)  # 对 heads 维度取平均
        data3 = data3[:seq_len, :seq_len]
    elif attention_matrix.ndim == 3:
        # [H, N, N] -> 对所有 heads 取平均
        data3 = attention_matrix.mean(axis=0)[:seq_len, :seq_len]
    else:
        data3 = attention_matrix[:seq_len, :seq_len]

    # 计算每个query位置的平均注意力权重
    mean_attn_per_query = to_numpy(data3.mean(axis=1))
    std_attn_per_query = to_numpy(data3.std(axis=1))
    x = np.arange(len(mean_attn_per_query))

    ax3.fill_between(x, mean_attn_per_query - std_attn_per_query,
                     mean_attn_per_query + std_attn_per_query, alpha=0.3)
    ax3.plot(x, mean_attn_per_query, 'b-', linewidth=1.5)
    ax3.set_title('Attention Weight Distribution\nby Query Position', fontsize=12)
    ax3.set_xlabel('Query Position ')
    ax3.set_ylabel('Mean Attention Weight')
    ax3.grid(True, alpha=0.3)

    # 保存子图3为 PDF
    _save_subplot_as_pdf(
        {'x': x, 'mean': mean_attn_per_query, 'std': std_attn_per_query},
        os.path.join(output_dir, f'{base_name}_distribution.pdf'),
        'Attention Weight Distribution by Query Position',
        'Query Position',
        'Mean Attention Weight',
        is_heatmap=False
    )

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved PNG: {save_path}")
    print(f"Saved 3 PDF files: head1.pdf, multihead_avg.pdf, distribution.pdf")


def visualize_multi_head_attention(attentions, seq_len, save_path):
    """
    绘制多头注意力的详细对比图
    """
    if not attentions:
        print("No attention weights to visualize")
        return
    
    # 获取第一个batch的数据
    attn_data = attentions[0]['attentions']
    
    # attn_data: list of [B, N, H, N] for TSSA
    first_layer_attn = attn_data[0]  # [B, N, H, N]
    
    # 正确解析维度：从注意力矩阵推断实际的 N（变量数）
    B, actual_N, H, N2 = first_layer_attn.shape
    assert actual_N == N2, f"Attention matrix must be square, got {actual_N} != {N2}"
    
    print(f"DEBUG: Attention matrix shape = {first_layer_attn.shape}")
    print(f"  - Batch size B = {B}")
    print(f"  - Number of variables/tokens N = {actual_N}")
    print(f"  - Number of heads H = {H}")
    
    # 计算每个head的注意力: [B, N, H, N] -> [H, N, N]
    # 注意：不取平均，保留每个head的注意力矩阵
    head_attentions = first_layer_attn[0, :actual_N, :, :actual_N].transpose(0, 1)  # [H, N, N]
    
    # 移动到 CPU 并转换为 numpy
    head_attentions = to_numpy(head_attentions)
    
    print(f"DEBUG: head_attentions shape = {head_attentions.shape}")
    
    # 创建图形
    n_heads = head_attentions.shape[0]
    n_cols = min(4, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(n_rows, -1)
    axes = axes.flatten()
    
    # 自定义颜色映射
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    
    for h in range(n_heads):
        ax = axes[h]
        im = ax.imshow(head_attentions[h], cmap='viridis', aspect='auto',
                      interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f'Head {h+1}', fontsize=11)
        ax.set_xlabel('Key Position' if h >= (n_rows-1)*n_cols else '')
        ax.set_ylabel('Query Position' if h % n_cols == 0 else '')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 隐藏多余的子图
    for h in range(n_heads, len(axes)):
        axes[h].axis('off')
    
    fig.suptitle('Multi-Head Attention Weights (Layer 1)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def visualize_attention_advanced(attentions, seq_len, save_path):
    """
    高级注意力可视化：包含更多分析维度
    """
    if not attentions:
        print("No attention weights to visualize")
        return

    attn_data = attentions[0]['attentions']
    first_layer_attn = attn_data[0]  # [B, N, H, N]
    B, N, H, N2 = first_layer_attn.shape

    print(f"DEBUG: visualize_attention_advanced - actual N={N}, H={H}")

    # 使用实际的注意力维度 N
    actual_seq_len = N

    # 收集所有层和头的注意力，并立即转换为 numpy
    all_attns_list = []
    for layer_attn in attn_data:
        # [B, N, H, N] = [1, 11, 8, 11]
        # layer_attn[0, :, :, :] = [N, H, N] = [11, 8, 11]
        # mean(dim=1) 对 H 取平均 -> [N, N] = [11, 11]
        layer_attn_avg = layer_attn[0, :, :, :].mean(dim=1)  # [N, N]
        all_attns_list.append(to_numpy(layer_attn_avg))

    all_attns = np.array(all_attns_list)  # [n_layers, N, N]
    n_layers = all_attns.shape[0]

    # 获取输出目录和基础名称
    output_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]

    # 保存每个子图为单独的 PDF
    print(f"\n[Saving individual PDFs for attention_analysis]")

    # 1. 各层注意力对比热力图
    for l in range(min(3, n_layers)):
        fig, ax = plt.subplots(figsize=(8, 6))
        layer_attn = all_attns[l]  # [N, N]
        im = ax.imshow(layer_attn, cmap='YlOrRd', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f'Layer {l+1} Attention Weight Matrix', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # 设置刻度
        tick_step = max(1, N // 8)
        tick_positions = np.arange(0, N, tick_step)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels([str(i + 1) for i in tick_positions])
        ax.set_yticklabels([str(i + 1) for i in tick_positions])

        plt.tight_layout()
        pdf_path = os.path.join(output_dir, f'{base_name}_layer{l+1}_heatmap.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"  Saved: {pdf_path}")

    # 2. 跨层注意力变化图
    fig, ax = plt.subplots(figsize=(8, 6))
    layer_diff = []
    for l in range(1, n_layers):
        diff = np.abs(all_attns[l] - all_attns[l-1]).mean()
        layer_diff.append(diff)

    x_evolution = np.arange(1, n_layers) if n_layers > 1 else np.array([1])
    if len(layer_diff) > 0:
        ax.plot(x_evolution, layer_diff, 'o-', color='blue', alpha=0.8, linewidth=2, markersize=8)
    else:
        ax.text(0.5, 0.5, 'Only 1 layer\nNo evolution data', ha='center', va='center', fontsize=14)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Attention Change (L1 Distance)', fontsize=12)
    ax.set_title('Attention Evolution Across Layers', fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    if n_layers > 1:
        ax.set_xticks(x_evolution)

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{base_name}_layer_evolution.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pdf_path}")

    # 3. 对角线注意力强度图
    fig, ax = plt.subplots(figsize=(8, 6))
    for l in range(min(3, n_layers)):
        layer_attn = all_attns[l]
        diag_values = []
        for offset in range(-N//4, N//4 + 1):
            diag = np.diag(layer_attn, k=offset)
            diag_values.append(diag.mean())
        diag_values = np.array(diag_values)
        offsets = np.arange(-N//4, N//4 + 1)
        ax.plot(offsets, diag_values, 'o-', label=f'Layer {l+1}', alpha=0.8, linewidth=2)

    ax.set_xlabel('Diagonal Offset', fontsize=12)
    ax.set_ylabel('Mean Attention Weight', fontsize=12)
    ax.set_title('Diagonal Attention Strength Analysis', fontsize=14, fontweight='bold', pad=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{base_name}_diagonal_strength.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pdf_path}")

    # 4. Query位置注意力集中度（熵）
    fig, ax = plt.subplots(figsize=(8, 6))
    entropy_per_query = []
    for l in range(n_layers):
        layer_attn = all_attns[l]
        attn_probs = layer_attn / (layer_attn.sum(axis=-1, keepdims=True) + 1e-8)
        entropy = -np.sum(attn_probs * np.log(attn_probs + 1e-8), axis=-1)
        entropy_per_query.append(entropy)
    entropy_per_query = np.array(entropy_per_query).mean(axis=0)

    x = np.arange(actual_seq_len)
    ax.plot(x, entropy_per_query, 'b-', linewidth=1.5)
    ax.fill_between(x, entropy_per_query, alpha=0.3, color='blue')
    ax.set_xlabel('Query Position', fontsize=12)
    ax.set_ylabel('Attention Entropy (nats)', fontsize=12)
    ax.set_title('Attention Concentration by Query Position\n(Higher = More Uniform Distribution)', fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{base_name}_entropy_query.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pdf_path}")

    # 5. Key位置被关注程度
    fig, ax = plt.subplots(figsize=(8, 6))
    attention_coverage = all_attns.mean(axis=0)
    coverage_per_key = attention_coverage.sum(axis=0)

    ax.bar(x, coverage_per_key, color='steelblue', alpha=0.7)
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Total Attention Received', fontsize=12)
    ax.set_title('Attention Coverage per Key Position', fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{base_name}_coverage_key.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pdf_path}")

    # 6. Top-K 注意力稀疏性分析
    fig, ax = plt.subplots(figsize=(8, 6))
    k_values = [1, 3, 5, 10, 20]
    sparsity_coverage = []
    for l in range(n_layers):
        layer_attn = all_attns[l]
        sorted_attn = np.sort(layer_attn, axis=-1)[:, -k_values[-1]:]
        for k in k_values:
            coverage = sorted_attn[:, -k:].sum(axis=-1).mean()
            sparsity_coverage.append((l+1, k, coverage))

    x_layers = np.arange(1, n_layers + 1)
    width = 0.15
    colors_bar = plt.cm.viridis(np.linspace(0, 1, len(k_values)))

    for i, k in enumerate(k_values):
        coverages = [cov for l, kk, cov in sparsity_coverage if kk == k]
        ax.bar(x_layers + i*width - 1.5*width, coverages, width,
               label=f'Top-{k}', color=colors_bar[i], alpha=0.8)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Attention Coverage', fontsize=12)
    ax.set_title('Top-K Attention Coverage by Layer', fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks(x_layers)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{base_name}_topk_sparsity.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pdf_path}")

    # 7. 长期vs短期注意力模式
    fig, ax = plt.subplots(figsize=(8, 6))
    layer_attn = all_attns[0]
    time_distances = np.arange(-N//2, N//2 + 1)
    attn_by_distance = []

    for d in time_distances:
        if abs(d) < N:
            diag = np.diag(layer_attn, k=d)
            attn_by_distance.append(diag.mean())
        else:
            attn_by_distance.append(0)

    ax.plot(time_distances, attn_by_distance, 'r-', linewidth=2)
    ax.fill_between(time_distances, attn_by_distance, alpha=0.3, color='red')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Distance (Key - Query)', fontsize=12)
    ax.set_ylabel('Mean Attention Weight', fontsize=12)
    ax.set_title('Attention Weight vs Time Distance\n(Layer 1)', fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{base_name}_time_distance.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pdf_path}")

    # 8. 注意力权重直方图
    fig, ax = plt.subplots(figsize=(8, 6))
    all_attn_values = all_attns.flatten()
    ax.hist(all_attn_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Attention Weight', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Attention Weight Distribution\n(All Layers Combined)', fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)

    # 添加统计信息
    mean_val = all_attn_values.mean()
    std_val = all_attn_values.std()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_val:.3f}')
    ax.legend()

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{base_name}_histogram.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pdf_path}")

    # 9. 统计总结
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    diag_mean = np.diag(layer_attn).mean()
    top1_coverage = [cov for l, k, cov in sparsity_coverage if k == 1][0]

    stats_text = f"""Attention Statistics Summary
{'='*50}

Model Architecture:
  Total Layers: {n_layers}
  Total Heads: {H}
  Attention Dimension (N): {actual_seq_len}

Attention Weight Statistics:
  Mean: {all_attns.mean():.4f}
  Std: {all_attns.std():.4f}
  Min: {all_attns.min():.4f}
  Max: {all_attns.max():.4f}

Diagonal Attention (Self-Focus):
  Mean: {diag_mean:.4f}

Top-K Coverage:
  Top-1:  {[cov for l, k, cov in sparsity_coverage if k == 1][0]:.2%}
  Top-3:  {[cov for l, k, cov in sparsity_coverage if k == 3][0]:.2%}
  Top-5:  {[cov for l, k, cov in sparsity_coverage if k == 5][0]:.2%}

Layer Evolution:
  Max Change: {max(layer_diff) if layer_diff else 0:.4f}
  Min Change: {min(layer_diff) if layer_diff else 0:.4f}
"""
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{base_name}_statistics.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pdf_path}")

    # 创建组合 PNG（原有功能）
    fig = plt.figure(figsize=(20, 16))

    # 1. 各层注意力对比（第一行）
    for l in range(min(3, n_layers)):
        ax = fig.add_subplot(4, 3, l+1)
        layer_attn = all_attns[l]
        im = ax.imshow(layer_attn, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        ax.set_title(f'Layer {l+1} Attention', fontsize=11)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # 2. 跨层注意力变化（第二行左2）
    ax = fig.add_subplot(4, 3, 4)
    x_evolution = np.arange(1, n_layers) if n_layers > 1 else np.array([1])
    if len(layer_diff) > 0:
        ax.plot(x_evolution, layer_diff, 'o-', color='blue', alpha=0.8, linewidth=2)
    else:
        ax.text(0.5, 0.5, 'Only 1 layer\nNo evolution data', ha='center', va='center', fontsize=12)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Attention Change (L1 Distance)')
    ax.set_title('Attention Evolution Across Layers', fontsize=11)
    ax.grid(True, alpha=0.3)

    # 3. 注意力对角线强度分析（第二行右1）
    ax = fig.add_subplot(4, 3, 6)
    for l in range(min(3, n_layers)):
        layer_attn = all_attns[l]
        diag_values = []
        for offset in range(-N//4, N//4 + 1):
            diag = np.diag(layer_attn, k=offset)
            diag_values.append(diag.mean())
        diag_values = np.array(diag_values)
        offsets = np.arange(-N//4, N//4 + 1)
        ax.plot(offsets, diag_values, 'o-', label=f'Layer {l+1}', alpha=0.8)
    ax.set_xlabel('Diagonal Offset')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('Diagonal Attention Strength', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Query位置注意力集中度分析（第三行）
    ax = fig.add_subplot(4, 3, 7)
    x_query = np.arange(actual_seq_len)
    ax.plot(x_query, entropy_per_query, 'b-', linewidth=1.5)
    ax.fill_between(x_query, entropy_per_query, alpha=0.3)
    ax.set_xlabel('Query Position')
    ax.set_ylabel('Attention Entropy (nats)')
    ax.set_title('Attention Concentration (Higher = More Uniform)', fontsize=11)
    ax.grid(True, alpha=0.3)

    # 5. Key位置被关注程度（第三行中间）
    ax = fig.add_subplot(4, 3, 8)
    ax.bar(x_query, coverage_per_key, color='steelblue', alpha=0.7)
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Total Attention Received')
    ax.set_title('Attention Coverage per Key Position', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # 6. 注意力稀疏性分析（第三行右1）
    ax = fig.add_subplot(4, 3, 9)
    for i, k in enumerate(k_values):
        coverages = [cov for l, kk, cov in sparsity_coverage if kk == k]
        ax.bar(x_layers + i*width - 1.5*width, coverages, width,
               label=f'Top-{k}', color=colors_bar[i], alpha=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Attention Coverage')
    ax.set_title('Top-K Attention Coverage', fontsize=11)
    ax.set_xticks(x_layers)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # 7. 长期vs短期注意力模式（第四行）
    ax = fig.add_subplot(4, 3, 10)
    ax.plot(time_distances, attn_by_distance, 'r-', linewidth=2)
    ax.fill_between(time_distances, attn_by_distance, alpha=0.3, color='red')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Distance (Key - Query)')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('Attention vs Time Distance', fontsize=11)
    ax.grid(True, alpha=0.3)

    # 8. 注意力权重直方图
    ax = fig.add_subplot(4, 3, 11)
    ax.hist(all_attn_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Attention Weight')
    ax.set_ylabel('Frequency')
    ax.set_title('Attention Weight Distribution', fontsize=11)
    ax.grid(True, alpha=0.3)

    # 9. 总结统计
    ax = fig.add_subplot(4, 3, 12)
    ax.axis('off')

    stats_text = f"""Attention Statistics Summary
{'='*40}

Total Layers: {n_layers}
Total Heads: {H}
Sequence Length (Attention N): {actual_seq_len}

Attention Weight Statistics:
  Mean: {all_attns.mean():.4f}
  Std: {all_attns.std():.4f}
  Min: {all_attns.min():.4f}
  Max: {all_attns.max():.4f}

Diagonal Attention (self-focus):
  Mean: {diag_mean:.4f}

Top-1 Attention Coverage:
  {top1_coverage:.2%}
"""
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Comprehensive Query-Key Attention Analysis\n(iTransformer Long Sequence Prediction)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved PNG: {save_path}")
    print(f"Saved {1 + min(3, n_layers) + 7} PDF files to: {output_dir}")
    layer_attn = all_attns[0]  # [N, N]
    time_distances = np.arange(-N//2, N//2 + 1)
    attn_by_distance = []
    
    for d in time_distances:
        if abs(d) < N:
            diag = np.diag(layer_attn, k=d)
            attn_by_distance.append(diag.mean())
        else:
            attn_by_distance.append(0)
    
    ax.plot(time_distances, attn_by_distance, 'r-', linewidth=2)
    ax.fill_between(time_distances, attn_by_distance, alpha=0.3, color='red')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Distance (Key - Query)')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('Attention vs Time Distance', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 8. 注意力权重直方图
    ax = fig.add_subplot(4, 3, 11)
    all_attn_values = all_attns.flatten()
    ax.hist(all_attn_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Attention Weight')
    ax.set_ylabel('Frequency')
    ax.set_title('Attention Weight Distribution', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 9. 总结统计
    ax = fig.add_subplot(4, 3, 12)
    ax.axis('off')
    
    diag_mean = np.diag(layer_attn).mean()
    top1_coverage = [cov for l, k, cov in sparsity_coverage if k == 1][0]
    
    stats_text = f"""Attention Statistics Summary
{'='*40}

Total Layers: {n_layers}
Total Heads: {H}
Sequence Length (Attention N): {actual_seq_len}

Attention Weight Statistics:
  Mean: {all_attns.mean():.4f}
  Std: {all_attns.std():.4f}
  Min: {all_attns.min():.4f}
  Max: {all_attns.max():.4f}

Diagonal Attention (self-focus):
  Mean: {diag_mean:.4f}
  
Top-1 Attention Coverage:
  {top1_coverage:.2%}
"""
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Comprehensive Query-Key Attention Analysis\n(iTransformer Long Sequence Prediction)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")



# ============================================================
# 保存单个热力图为 PDF 文件的辅助函数
# ============================================================

def save_single_heatmap_pdf(data, save_path, title="Attention Heatmap",
                             xlabel="Key Position", ylabel="Query Position",
                             figsize=(8, 6), cmap='viridis'):
    """保存单个热力图为 PDF 文件"""
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    n = data.shape[0]
    tick_step = max(1, n // 8)
    tick_positions = np.arange(0, n, tick_step)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([str(i+1) for i in tick_positions])
    ax.set_yticklabels([str(i+1) for i in tick_positions])

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved PDF: {save_path}")


def save_individual_layer_heatmaps(attentions, output_dir, seq_len=None):
    """将每一层的注意力热力图单独保存为 PDF 文件"""
    if not attentions:
        print("No attention weights to save")
        return

    layer_dir = os.path.join(output_dir, 'layer_heatmaps')
    os.makedirs(layer_dir, exist_ok=True)

    attn_data = attentions[0]['attentions']
    n_layers = len(attn_data)

    print(f"\n[Saving Layer Heatmaps to PDF]")
    print(f"  Output directory: {layer_dir}")

    for layer_idx in range(n_layers):
        layer_attn = attn_data[layer_idx]
        if isinstance(layer_attn, torch.Tensor):
            layer_attn = to_numpy(layer_attn)

        B, N, H, N2 = layer_attn.shape
        assert N == N2

        if seq_len is not None:
            N = min(N, seq_len)

        avg_attn = layer_attn[0, :N, :, :N].mean(axis=0)

        save_path = os.path.join(layer_dir, f'layer{layer_idx+1}_average_attention.pdf')
        save_single_heatmap_pdf(
            avg_attn, save_path,
            title=f"Layer {layer_idx+1}: Average Multi-Head Attention",
            xlabel="Key Position",
            ylabel="Query Position"
        )


def save_individual_head_heatmaps(attentions, output_dir, seq_len=None):
    """将每个头的注意力热力图单独保存为 PDF 文件"""
    if not attentions:
        print("No attention weights to save")
        return

    head_dir = os.path.join(output_dir, 'head_heatmaps')
    os.makedirs(head_dir, exist_ok=True)

    attn_data = attentions[0]['attentions']
    first_layer_attn = attn_data[0]

    if isinstance(first_layer_attn, torch.Tensor):
        first_layer_attn = to_numpy(first_layer_attn)

    B, N, H, N2 = first_layer_attn.shape
    assert N == N2

    if seq_len is not None:
        N = min(N, seq_len)

    # 提取每个头的注意力: [B, N, H, N] -> [N, H, N] -> [H, N, N]
    # 使用 np.moveaxis 确保正确的轴转换
    sliced = first_layer_attn[0, :N, :, :N]  # [N, H, N]
    head_attentions = np.moveaxis(sliced, 1, 0)  # [H, N, N]

    print(f"\n[Saving Head Heatmaps to PDF]")
    print(f"  Output directory: {head_dir}")
    print(f"  Number of heads: {H}")
    print(f"  head_attentions shape: {head_attentions.shape}")

    for h in range(H):
        head_attn = head_attentions[h]
        save_path = os.path.join(head_dir, f'layer1_head{h+1}_attention.pdf')
        save_single_heatmap_pdf(
            head_attn, save_path,
            title=f"Layer 1 - Head {h+1}: Attention Weights",
            xlabel="Key Position",
            ylabel="Query Position"
        )


def save_advanced_analysis_pdfs(attentions, output_dir, seq_len=None):
    """保存高级分析图表为 PDF 文件"""
    if not attentions:
        print("No attention weights to analyze")
        return

    analysis_dir = os.path.join(output_dir, 'analysis_pdfs')
    os.makedirs(analysis_dir, exist_ok=True)

    attn_data = attentions[0]['attentions']
    n_layers = len(attn_data)

    print(f"\n[Saving Analysis PDFs]")
    print(f"  Output directory: {analysis_dir}")

    all_attns_list = []
    for layer_attn in attn_data:
        if isinstance(layer_attn, torch.Tensor):
            layer_attn = to_numpy(layer_attn)
        layer_avg = layer_attn[0, :, :, :].mean(axis=0)
        all_attns_list.append(layer_avg)

    all_attns = np.array(all_attns_list)
    N = all_attns.shape[1]

    # 1. 对角线注意力强度图
    fig, ax = plt.subplots(figsize=(10, 6))
    for l in range(min(3, n_layers)):
        layer_attn = all_attns[l]
        diag_values = [np.diag(layer_attn, k=offset).mean() for offset in range(-N//4, N//4 + 1)]
        ax.plot(list(range(-N//4, N//4 + 1)), diag_values, 'o-', label=f'Layer {l+1}', linewidth=2)
    ax.set_xlabel('Diagonal Offset', fontsize=12)
    ax.set_ylabel('Mean Attention Weight', fontsize=12)
    ax.set_title('Diagonal Attention Strength Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(analysis_dir, 'diagonal_attention_strength.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    # 2. 熵分析图
    fig, ax = plt.subplots(figsize=(10, 6))
    entropy_per_layer = []
    for l in range(n_layers):
        layer_attn = all_attns[l]
        attn_probs = layer_attn / (layer_attn.sum(axis=-1, keepdims=True) + 1e-8)
        entropy = -np.sum(attn_probs * np.log(attn_probs + 1e-8), axis=-1)
        entropy_per_layer.append(entropy.mean())
    ax.bar(range(1, n_layers + 1), entropy_per_layer, color='steelblue', alpha=0.7)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Attention Entropy', fontsize=12)
    ax.set_title('Attention Entropy by Layer', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_path = os.path.join(analysis_dir, 'attention_entropy_by_layer.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    # 3. 跨层注意力变化图
    fig, ax = plt.subplots(figsize=(10, 6))
    layer_diff = [np.abs(all_attns[l] - all_attns[l-1]).mean() for l in range(1, n_layers)]
    ax.plot(range(2, n_layers + 1), layer_diff, 'o-', color='crimson', linewidth=2, markersize=8)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Attention Change (L1 Distance)', fontsize=12)
    ax.set_title('Attention Evolution Across Layers', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(analysis_dir, 'attention_evolution.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    # 4. 跨层注意力对比图
    n_show = min(4, n_layers)
    fig, axes = plt.subplots(1, n_show, figsize=(5*n_show, 4))
    if n_show == 1:
        axes = [axes]
    for l in range(n_show):
        im = axes[l].imshow(all_attns[l], cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        axes[l].set_title(f'Layer {l+1}', fontsize=12)
        plt.colorbar(im, ax=axes[l], shrink=0.8)
    fig.suptitle('Cross-Layer Attention Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(analysis_dir, 'cross_layer_comparison.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    print(f"\n  Total PDFs saved: 4")


def save_all_heatmaps_as_pdfs(attentions, output_dir, seq_len=None):
    """保存所有热力图为单独的 PDF 文件"""
    print("\n" + "="*60)
    print("Saving Individual PDF Heatmaps")
    print("="*60)
    save_individual_layer_heatmaps(attentions, output_dir, seq_len)
    save_individual_head_heatmaps(attentions, output_dir, seq_len)
    save_advanced_analysis_pdfs(attentions, output_dir, seq_len)
    print("\n" + "="*60)
    print("PDF Export Complete!")
    print(f"All PDFs saved to: {output_dir}")
    print("="*60)


def main(args):
    """主函数"""
    print("="*60)
    print("Query-Key Attention Visualization for iTransformer")
    print("="*60)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型和数据
    print("\n[1/4] Loading model and data...")
    try:
        model, test_loader, args = load_model_and_data(args)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 提取注意力权重
    print("\n[2/4] Extracting attention weights...")
    attentions = extract_attention_weights(model, test_loader, device, args, max_batches=args.max_batches)
    
    if not attentions:
        print("No attention weights extracted!")
        return
    
    # 获取序列长度
    sample_batch = attentions[0]['input']
    seq_len = min(sample_batch.shape[1], args.seq_len)
    
    # 绘制基本热力图
    print("\n[3/4] Generating attention heatmaps...")
    attn_data = attentions[0]['attentions']
    
    # 基本热力图
    save_path = os.path.join(output_dir, 'attention_heatmap.png')
    visualize_attention_heatmap(
        to_numpy(attn_data[0]), 
        seq_len, 
        save_path,
        title="Query-Key Attention Weights (iTransformer Long Sequence Prediction)"
    )
    
    # 多头注意力对比
    save_path = os.path.join(output_dir, 'multi_head_attention.png')
    visualize_multi_head_attention(attentions, seq_len, save_path)
    
    # 高级分析
    save_path = os.path.join(output_dir, 'attention_analysis.png')
    visualize_attention_advanced(attentions, seq_len, save_path)

    # ============================================================
    # 新增：保存单独的 PDF 文件
    # ============================================================
    print("\n[4/5] Saving individual PDF heatmaps...")
    save_all_heatmaps_as_pdfs(attentions, output_dir, seq_len)

    print("\n[5/5] Visualization complete!")
    print(f"All outputs saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attention Visualization')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    
    # 项目根目录（用于导入模型）
    parser.add_argument('--project_root', type=str, default='.',
                        help='Project root directory')
    
    # 数据参数
    parser.add_argument('--data', type=str, default='ETTh1',
                        help='Dataset name')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/',
                        help='Root path of data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv',
                        help='Data file name')
    parser.add_argument('--features', type=str, default='M',
                        help='Forecasting task: M (multivariate), S (univariate)')
    parser.add_argument('--target', type=str, default='OT',
                        help='Target feature')
    parser.add_argument('--seq_len', type=int, default=96,
                        help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=48,
                        help='Start token length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction length')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='Number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='Time feature encoding')
    parser.add_argument('--activation', type=str, default='gelu',
                        help='Activation function')
    
    # 其他参数
    parser.add_argument('--freq', type=str, default='h',
                        help='Time frequency: s, t, h, d, b, w, m')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='Checkpoints path')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='Use GPU')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='Use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='Device IDs for multi-GPU')
    parser.add_argument('--use_norm', type=bool, default=True,
                        help='Use normalization')
    parser.add_argument('--max_batches', type=int, default=5,
                        help='Maximum batches to process')
    parser.add_argument('--output_dir', type=str, default='./attention_results',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=2023,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # 设置GPU
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.device = torch.device('cuda' if args.use_gpu else 'cpu')
    
    # 设置项目根目录
    if args.project_root == '.':
        args.project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 自动从检查点路径解析模型参数
    import re
    checkpoint_name = os.path.basename(os.path.dirname(args.model_path))
    print(f"Parsing parameters from checkpoint: {checkpoint_name}")
    
    # ETTh1 数据集固定参数
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    
    # 从前缀提取 seq_len 和 pred_len（格式: ETTh1_96_720_）
    # 注意：检查点文件名可能有bug，使用前缀更可靠
    prefix_match = re.match(r'(\w+)_(\d+)_(\d+)_', checkpoint_name)
    if prefix_match:
        # 格式: data_inputLen_outputLen_... 
        # input_len -> seq_len, output_len -> pred_len (通常是预测长度)
        prefix_input = int(prefix_match.group(2))
        prefix_output = int(prefix_match.group(3))
        print(f"  Prefix found: input={prefix_input}, output={prefix_output}")
    else:
        prefix_input = 96  # 默认值
        prefix_output = 720
        print(f"  No prefix found, using defaults")
    
    # 解析模型部分参数
    # 提取模型名称部分（去掉数据名前缀）
    model_name_match = re.search(r'(iTransformer_.+)', checkpoint_name)
    if model_name_match:
        model_part = model_name_match.group(1)
    else:
        model_part = checkpoint_name
    
    # 解析 d_model (dm 或 _dm)
    dm_match = re.search(r'_dm(\d+)[_\]]', model_part)
    if not dm_match:
        dm_match = re.search(r'_dm(\d+)', model_part)
    args.d_model = int(dm_match.group(1)) if dm_match else 512
    print(f"  d_model from filename: {args.d_model}")
    
    # 检查 d_model 是否需要修正（基于 state_dict 中的常见值）
    # 如果 d_model=8 但其他参数暗示更大的模型，可能需要修正
    if args.d_model == 8:
        # 检查 dl512 是否暗示 d_ff=512，间接暗示 d_model=512
        if '_dl512_' in model_part:
            print(f"  WARNING: d_model in filename=8, but dl512 suggests d_model=512")
            # 使用 dl 的值来推断 d_model
            args.d_model = 512
            print(f"  Corrected d_model to: {args.d_model}")
    
    # 解析 n_heads
    nh_match = re.search(r'_nh(\d+)[_\]]', model_part)
    if not nh_match:
        nh_match = re.search(r'_nh(\d+)', model_part)
    args.n_heads = int(nh_match.group(1)) if nh_match else 8
    
    # 解析 e_layers
    el_match = re.search(r'_el(\d+)[_\]]', model_part)
    args.e_layers = int(el_match.group(1)) if el_match else 2
    
    # 解析 d_ff (优先使用 dl，因为 df 可能被误解)
    dl_match = re.search(r'_dl(\d+)[_\]]', model_part)
    df_match = re.search(r'_df(\d+)[_\]]', model_part)
    
    if dl_match:
        args.d_ff = int(dl_match.group(1))
        print(f"  d_ff from dl: {args.d_ff}")
    elif df_match:
        d_ff_candidate = int(df_match.group(1))
        # 如果 df=1 但 dl=512 存在，可能 df 是 bug
        if d_ff_candidate == 1 and '_dl512_' in model_part:
            print(f"  WARNING: df=1 detected, but dl512 suggests d_ff=512")
            args.d_ff = 512
        else:
            args.d_ff = d_ff_candidate
    else:
        args.d_ff = 512
    
    # 解析 seq_len（使用前缀更可靠）
    args.seq_len = prefix_input
    
    # 解析 pred_len（使用前缀更可靠）
    args.pred_len = prefix_output
    
    # 解析 label_len
    ll_match = re.search(r'_ll(\d+)[_\]]', model_part)
    if ll_match:
        args.label_len = int(ll_match.group(1))
    else:
        args.label_len = args.seq_len // 2
    
    print(f"\n  Final parsed parameters:")
    print(f"    seq_len: {args.seq_len}")
    print(f"    pred_len: {args.pred_len}")
    print(f"    label_len: {args.label_len}")
    print(f"    d_model: {args.d_model}")
    print(f"    n_heads: {args.n_heads}")
    print(f"    e_layers: {args.e_layers}")
    print(f"    d_ff: {args.d_ff}")
    
    main(args)