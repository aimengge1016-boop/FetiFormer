"""
FullAttention 注意力权重可视化脚本
专门用于 iTransformer原始.py (标准 FullAttention 模型)
"""

import os
import re
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def to_numpy(tensor):
    """将 CUDA tensor 转换为 numpy 数组"""
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.numpy()
    return tensor


def load_model_and_data(args):
    """加载模型和数据"""
    from data_provider.data_factory import data_provider

    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Loading checkpoint: {args.model_path}")
    print(f"Using device: {device}")

    # 先加载检查点到 CPU 以获取参数
    checkpoint = torch.load(args.model_path, map_location='cpu')

    # 从检查点推断模型参数
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
        print(f"  From projector: pred_len={actual_pred_len}, d_model={actual_d_model}")

    # enc_embedding.value_embedding.weight 形状为 [d_model, seq_len]
    if 'enc_embedding.value_embedding.weight' in checkpoint:
        emb_shape = checkpoint['enc_embedding.value_embedding.weight'].shape
        actual_seq_len = int(emb_shape[1])
        print(f"  From embedding: seq_len={actual_seq_len}")

    # n_heads 从 query_projection 推断
    if 'encoder.attn_layers.0.attention.query_projection.weight' in checkpoint:
        q_shape = checkpoint['encoder.attn_layers.0.attention.query_projection.weight'].shape
        actual_n_heads = q_shape[0] // (actual_d_model // 8)  # 假设 d_keys = d_model // n_heads
        print(f"  From projection: n_heads={actual_n_heads}")

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

    # 更新 args
    args.d_model = actual_d_model
    args.pred_len = actual_pred_len
    args.seq_len = actual_seq_len
    args.n_heads = actual_n_heads
    args.e_layers = actual_e_layers
    args.d_ff = actual_d_ff

    # 添加模型需要的属性
    args.output_attention = True
    args.use_norm = True
    args.use_time2vec = True
    args.dropout = 0.1
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.freq = 'h'
    args.num_workers = 0
    args.batch_size = 8
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    args.class_strategy = 'average'
    args.factor = 5
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

    # 导入标准版本的模型
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "iTransformer_standard",
        os.path.join(args.project_root, "model", "iTransformer.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.Model(args).float()
    model = model.to(device)

    # 将检查点张量移动到设备上
    checkpoint_on_device = {k: v.to(device) for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint_on_device, strict=False)
    model.eval()

    _, test_loader = data_provider(args, flag='test')

    return model, test_loader, args


def extract_attention_weights(model, test_loader, device, args, max_batches=5):
    """从模型中提取注意力权重 (FullAttention 返回 [B, H, L, S])"""
    model.eval()
    all_attentions = []

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            if batch_idx >= max_batches:
                break

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device) if batch_x_mark is not None else None
            batch_y_mark = batch_y_mark.float().to(device) if batch_y_mark is not None else None

            original_output_attention = model.output_attention
            model.output_attention = True

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            try:
                outputs, attentions = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            except Exception as e:
                print(f"Error in forward pass: {e}")
                continue

            model.output_attention = original_output_attention

            # attentions: list of [B, H, L, S] for each layer
            all_attentions.append({
                'batch_idx': batch_idx,
                'attentions': attentions,  # list of [B, H, L, S]
                'input': to_numpy(batch_x)
            })

            print(f"Batch {batch_idx}: Extracted attention weights from {len(attentions)} layers")

    return all_attentions


def visualize_full_attention_heatmap(attention_matrix, seq_len, save_path, title="Attention Heatmap"):
    """
    绘制 FullAttention 注意力权重热力图

    Args:
        attention_matrix: [B, H, L, S] 形状 (FullAttention 格式)
        seq_len: 序列长度
        save_path: 保存路径
        title: 标题
    """

    # 获取输出目录和基础名称
    output_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]

    # 处理不同维度的输入
    if isinstance(attention_matrix, list):
        # 如果是列表，取第一个元素
        attention_matrix = attention_matrix[0]

    print(f"DEBUG: attention_matrix shape = {attention_matrix.shape}")

    # FullAttention 返回 [B, H, L, S]
    if attention_matrix.ndim == 4:
        B, H, L, S = attention_matrix.shape
        # 原始注意力 (取第一个 batch，第一个 head)
        data1 = attention_matrix[0, 0, :seq_len, :seq_len]
        # 多头平均 (跨 head 平均)
        data2 = attention_matrix[0, :, :seq_len, :seq_len].mean(axis=0)
        # 所有头的平均
        data3 = attention_matrix[0, :, :seq_len, :seq_len].mean(axis=(0, 1))
    else:
        data1 = data2 = data3 = attention_matrix[:seq_len, :seq_len]
        H = 1

    # 转换为 numpy (确保从 CUDA 移动到 CPU)
    data1 = to_numpy(data1)
    data2 = to_numpy(data2)
    data3 = to_numpy(data3)

    # 保存单个 PDF 文件
    print(f"\n[Saving individual PDFs for attention_heatmap]")

    # 1. Head 1 注意力热力图
    fig1, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data1, cmap='YlOrRd', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title('Head 1 Attention Weight Matrix', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)

    n = data1.shape[0]
    tick_step = max(1, n // 8)
    tick_positions = np.arange(0, n, tick_step)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([str(i + 1) for i in tick_positions])
    ax.set_yticklabels([str(i + 1) for i in tick_positions])

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{base_name}_head1.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pdf_path}")

    # 2. 多头平均注意力热力图
    fig2, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data2, cmap='YlOrRd', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f'Average Multi-Head Attention ({H} Heads)', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([str(i + 1) for i in tick_positions])
    ax.set_yticklabels([str(i + 1) for i in tick_positions])

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{base_name}_multihead_avg.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pdf_path}")

    # 3. 注意力权重分布直方图
    fig3, ax = plt.subplots(figsize=(8, 6))
    mean_attn = data3.mean()
    std_attn = data3.std()
    ax.hist(data3.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(mean_attn, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_attn:.3f}')
    ax.set_title('Attention Weight Distribution', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Attention Weight', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{base_name}_histogram.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pdf_path}")

    # 创建组合 PNG
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 第一个 Head 的注意力
    ax1 = axes[0]
    im1 = ax1.imshow(data1, cmap='YlOrRd', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax1.set_title(f'Head 1 Attention\n(Batch 1)', fontsize=12)
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 2. 多头平均注意力
    ax2 = axes[1]
    im2 = ax2.imshow(data2, cmap='YlOrRd', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax2.set_title(f'Multi-Head Average\n({H} Heads)', fontsize=12)
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # 3. 平均注意力权重分布
    ax3 = axes[2]
    ax3.hist(data3.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(mean_attn, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_attn:.3f}')
    ax3.set_title('Attention Weight Distribution', fontsize=12)
    ax3.set_xlabel('Attention Weight')
    ax3.set_ylabel('Frequency')
    ax3.legend()

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved PNG: {save_path}")
    print(f"Saved 3 PDF files to: {output_dir}")


def visualize_full_multi_head(attentions, seq_len, save_path):
    """
    绘制 FullAttention 多头注意力详细对比图

    attentions: list of [B, H, L, S] for each layer
    """
    if not attentions:
        print("No attention weights to visualize")
        return

    attn_data = attentions[0]['attentions']  # list of [B, H, L, S]
    first_layer_attn = attn_data[0]  # [B, H, L, S]

    if isinstance(first_layer_attn, torch.Tensor):
        first_layer_attn = to_numpy(first_layer_attn)

    B, H, L, S = first_layer_attn.shape
    print(f"DEBUG: FullAttention - B={B}, H={H}, L={L}, S={S}")

    # 使用实际的序列长度
    actual_len = min(L, seq_len)

    # 创建图形
    n_cols = min(4, H)
    n_rows = (H + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for h in range(H):
        ax = axes[h]
        data = first_layer_attn[0, h, :actual_len, :actual_len]
        im = ax.imshow(data, cmap='viridis', aspect='auto',
                      interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f'Head {h + 1}', fontsize=11)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # 隐藏多余的子图
    for h in range(H, len(axes)):
        axes[h].axis('off')

    fig.suptitle('FullAttention: Multi-Head Attention Weights (Layer 1)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


def visualize_full_attention_advanced(attentions, seq_len, save_path):
    """
    高级 FullAttention 可视化
    """
    if not attentions:
        print("No attention weights to visualize")
        return

    attn_data = attentions[0]['attentions']
    first_layer_attn = attn_data[0]

    if isinstance(first_layer_attn, torch.Tensor):
        first_layer_attn = to_numpy(first_layer_attn)

    B, H, L, S = first_layer_attn.shape
    print(f"DEBUG: advanced - B={B}, H={H}, L={L}, S={S}")
    print(f"DEBUG: Number of layers in attn_data: {len(attn_data)}")

    n_layers = len(attn_data)
    print(f"DEBUG: n_layers = {n_layers}")

    # 收集所有层的注意力 (确保每个都是 2D [L, S])
    all_attns_list = []
    for i, layer_attn in enumerate(attn_data):
        if isinstance(layer_attn, torch.Tensor):
            layer_attn = to_numpy(layer_attn)
        print(f"DEBUG: layer {i} shape before: {layer_attn.shape}")
        # [B, H, L, S] -> [L, S] (跨 batch 和 head 平均)
        if layer_attn.ndim == 4:
            layer_avg = layer_attn[0, :, :, :].mean(axis=(0, 1))
        elif layer_attn.ndim == 3:
            layer_avg = layer_attn[0, :, :].mean(axis=0)
        elif layer_attn.ndim == 2:
            layer_avg = layer_attn
        else:
            continue
        print(f"DEBUG: layer {i} shape after: {layer_avg.shape}")
        all_attns_list.append(layer_avg)

    if not all_attns_list:
        print("No valid attention data found")
        return

    all_attns = np.array(all_attns_list)
    print(f"DEBUG: all_attns shape = {all_attns.shape}")

    # 确保是 3D [n_layers, L, S]
    if all_attns.ndim == 2:
        all_attns = all_attns[np.newaxis, :, :]
    n_layers = all_attns.shape[0]

    # 获取输出目录和基础名称
    output_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]

    # 保存每个子图为单独的 PDF
    print(f"\n[Saving individual PDFs for attention_analysis]")

    # 1. 各层注意力热力图
    for l in range(min(3, n_layers)):
        fig, ax = plt.subplots(figsize=(8, 6))
        layer_attn = all_attns[l]
        im = ax.imshow(layer_attn, cmap='YlOrRd', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f'Layer {l + 1} Attention Weight Matrix', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8)

        n = layer_attn.shape[0]
        tick_step = max(1, n // 8)
        tick_positions = np.arange(0, n, tick_step)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels([str(i + 1) for i in tick_positions])
        ax.set_yticklabels([str(i + 1) for i in tick_positions])

        plt.tight_layout()
        pdf_path = os.path.join(output_dir, f'{base_name}_layer{l + 1}_heatmap.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"  Saved: {pdf_path}")

    # 2. 跨层注意力变化图
    fig, ax = plt.subplots(figsize=(8, 6))
    layer_diff = []
    for l in range(1, n_layers):
        diff = np.abs(all_attns[l] - all_attns[l - 1]).mean()
        layer_diff.append(diff)

    x_evo = np.arange(1, n_layers) if n_layers > 1 else np.array([1])
    if len(layer_diff) > 0:
        ax.plot(x_evo, layer_diff, 'o-', color='blue', alpha=0.8, linewidth=2, markersize=8)
    else:
        ax.text(0.5, 0.5, 'Only 1 layer\nNo evolution data', ha='center', va='center', fontsize=14)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Attention Change (L1)', fontsize=12)
    ax.set_title('Attention Evolution Across Layers', fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    if n_layers > 1:
        ax.set_xticks(x_evo)

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{base_name}_layer_evolution.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pdf_path}")

    # 3. 对角线强度分析图
    fig, ax = plt.subplots(figsize=(8, 6))
    for l in range(min(3, n_layers)):
        layer_attn = all_attns[l]
        L = layer_attn.shape[0]
        diag_values = []
        offsets = range(-L // 4, L // 4 + 1)
        for offset in offsets:
            diag = np.diag(layer_attn, k=offset)
            diag_values.append(diag.mean())
        ax.plot(list(offsets), diag_values, 'o-', label=f'Layer {l + 1}', alpha=0.8, linewidth=2)
    ax.set_xlabel('Diagonal Offset', fontsize=12)
    ax.set_ylabel('Mean Attention Weight', fontsize=12)
    ax.set_title('Diagonal Attention Strength Analysis', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{base_name}_diagonal_strength.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pdf_path}")

    # 4. 各层注意力熵柱状图
    fig, ax = plt.subplots(figsize=(8, 6))
    entropy_per_layer = []
    for l in range(n_layers):
        layer_attn = all_attns[l]
        attn_probs = layer_attn / (layer_attn.sum(axis=-1, keepdims=True) + 1e-8)
        entropy = -np.sum(attn_probs * np.log(attn_probs + 1e-8), axis=-1)
        entropy_per_layer.append(entropy.mean())
    entropy_per_layer = np.array(entropy_per_layer)

    x = np.arange(1, n_layers + 1)
    ax.bar(x, entropy_per_layer, color='steelblue', alpha=0.7)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Attention Entropy', fontsize=12)
    ax.set_title('Attention Entropy by Layer', fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(x)

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{base_name}_entropy_layer.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pdf_path}")

    # 5. 平均注意力热力图
    fig, ax = plt.subplots(figsize=(8, 6))
    avg_attn = all_attns.mean(axis=0)
    im = ax.imshow(avg_attn, cmap='YlOrRd', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title('Average Attention (All Layers)', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)

    n = avg_attn.shape[0]
    tick_step = max(1, n // 8)
    tick_positions = np.arange(0, n, tick_step)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([str(i + 1) for i in tick_positions])
    ax.set_yticklabels([str(i + 1) for i in tick_positions])

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{base_name}_avg_attention.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pdf_path}")

    # 6. Head 1 详细热力图
    fig, ax = plt.subplots(figsize=(8, 6))
    head1_attn = to_numpy(first_layer_attn[0, 0, :, :])
    im = ax.imshow(head1_attn, cmap='YlOrRd', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title('Head 1 Attention (Layer 1)', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)

    n = head1_attn.shape[0]
    tick_step = max(1, n // 8)
    tick_positions = np.arange(0, n, tick_step)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([str(i + 1) for i in tick_positions])
    ax.set_yticklabels([str(i + 1) for i in tick_positions])

    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'{base_name}_head1_detail.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {pdf_path}")

    # 7. 各 Head 注意力热力图
    for h_idx, h in enumerate(range(min(3, H))):
        fig, ax = plt.subplots(figsize=(8, 6))
        head_attn = to_numpy(first_layer_attn[0, h, :, :])
        im = ax.imshow(head_attn, cmap='viridis', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f'Head {h + 1} Attention (Layer 1)', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8)

        n = head_attn.shape[0]
        tick_step = max(1, n // 8)
        tick_positions = np.arange(0, n, tick_step)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels([str(i + 1) for i in tick_positions])
        ax.set_yticklabels([str(i + 1) for i in tick_positions])

        plt.tight_layout()
        pdf_path = os.path.join(output_dir, f'{base_name}_head{h + 1}.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"  Saved: {pdf_path}")

    # 创建组合 PNG
    fig = plt.figure(figsize=(20, 16))

    # 1. 各层注意力对比（第一行）
    for l in range(min(3, n_layers)):
        ax = fig.add_subplot(4, 3, l + 1)
        layer_attn = all_attns[l]
        im = ax.imshow(layer_attn, cmap='YlOrRd', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f'Layer {l + 1} Attention', fontsize=11)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # 2. 跨层注意力变化
    ax = fig.add_subplot(4, 3, 4)
    x_evo = np.arange(1, n_layers) if n_layers > 1 else np.array([1])
    if len(layer_diff) > 0:
        ax.plot(x_evo, layer_diff, 'o-', color='blue', alpha=0.8, linewidth=2)
    else:
        ax.text(0.5, 0.5, 'Only 1 layer', ha='center', va='center', fontsize=12)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Attention Change (L1)')
    ax.set_title('Attention Evolution Across Layers', fontsize=11)
    ax.grid(True, alpha=0.3)

    # 3. 对角线强度分析
    ax = fig.add_subplot(4, 3, 6)
    for l in range(min(3, n_layers)):
        layer_attn = all_attns[l]
        L = layer_attn.shape[0]
        diag_values = []
        offsets = range(-L // 4, L // 4 + 1)
        for offset in offsets:
            diag = np.diag(layer_attn, k=offset)
            diag_values.append(diag.mean())
        ax.plot(list(offsets), diag_values, 'o-', label=f'Layer {l + 1}', alpha=0.8)
    ax.set_xlabel('Diagonal Offset')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('Diagonal Attention Strength', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Query 位置注意力集中度
    ax = fig.add_subplot(4, 3, 7)
    ax.bar(x, entropy_per_layer, color='steelblue', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Attention Entropy')
    ax.set_title('Attention Entropy by Layer', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # 5. 注意力强度热力图
    ax = fig.add_subplot(4, 3, 8)
    avg_attn = all_attns.mean(axis=0)
    im = ax.imshow(avg_attn, cmap='YlOrRd', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title('Average Attention (All Layers)', fontsize=11)
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 6. 第一个 Head 的详细热力图
    ax = fig.add_subplot(4, 3, 9)
    head1_attn = to_numpy(first_layer_attn[0, 0, :, :])
    im = ax.imshow(head1_attn, cmap='YlOrRd', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title('Head 1 Attention (Layer 1)', fontsize=11)
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 7-9. 多头注意力 (取前3个 head)
    for h_idx, h in enumerate(range(min(3, H))):
        ax = fig.add_subplot(4, 3, 10 + h_idx)
        head_attn = to_numpy(first_layer_attn[0, h, :, :])
        im = ax.imshow(head_attn, cmap='viridis', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f'Head {h + 1} Attention', fontsize=11)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('FullAttention: Comprehensive Query-Key Attention Analysis\n(iTransformer Long Sequence Prediction)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved PNG: {save_path}")
    print(f"Saved {min(3, n_layers) + 6} PDF files to: {output_dir}")


# ============================================================
# PDF 保存函数
# ============================================================

def save_single_heatmap_pdf(data, save_path, title="Attention Heatmap",
                             xlabel="Key Position", ylabel="Query Position",
                             figsize=(8, 6), cmap='viridis'):
    """保存单个热力图为 PDF"""
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
    ax.set_xticklabels([str(i + 1) for i in tick_positions])
    ax.set_yticklabels([str(i + 1) for i in tick_positions])

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved PDF: {save_path}")


def save_fullattention_layer_pdfs(attentions, output_dir, seq_len=None):
    """保存 FullAttention 每层平均注意力为 PDF"""
    if not attentions:
        print("No attention weights")
        return

    layer_dir = os.path.join(output_dir, 'layer_heatmaps')
    os.makedirs(layer_dir, exist_ok=True)

    attn_data = attentions[0]['attentions']
    n_layers = len(attn_data)

    print(f"\n[Saving Layer Heatmaps to PDF]")
    for layer_idx in range(n_layers):
        layer_attn = attn_data[layer_idx]
        if isinstance(layer_attn, torch.Tensor):
            layer_attn = to_numpy(layer_attn)

        B, H, L, S = layer_attn.shape
        avg_attn = layer_attn[0, :, :min(L, seq_len), :min(S, seq_len)].mean(axis=0)

        save_path = os.path.join(layer_dir, f'layer{layer_idx + 1}_average_attention.pdf')
        save_single_heatmap_pdf(
            avg_attn, save_path,
            title=f"Layer {layer_idx + 1}: Average Multi-Head Attention",
            xlabel="Key Position",
            ylabel="Query Position"
        )


def save_fullattention_head_pdfs(attentions, output_dir, seq_len=None):
    """保存 FullAttention 每个 Head 注意力为 PDF"""
    if not attentions:
        print("No attention weights")
        return

    head_dir = os.path.join(output_dir, 'head_heatmaps')
    os.makedirs(head_dir, exist_ok=True)

    attn_data = attentions[0]['attentions']
    first_layer_attn = attn_data[0]

    if isinstance(first_layer_attn, torch.Tensor):
        first_layer_attn = to_numpy(first_layer_attn)

    B, H, L, S = first_layer_attn.shape

    print(f"\n[Saving Head Heatmaps to PDF]")
    for h in range(H):
        head_attn = first_layer_attn[0, h, :min(L, seq_len), :min(S, seq_len)]
        save_path = os.path.join(head_dir, f'layer1_head{h + 1}_attention.pdf')
        save_single_heatmap_pdf(
            head_attn, save_path,
            title=f"Layer 1 - Head {h + 1}: Attention Weights",
            xlabel="Key Position",
            ylabel="Query Position"
        )


def save_fullattention_pdfs(attentions, output_dir, seq_len=None):
    """保存所有 FullAttention 热力图为 PDF"""
    print("\n" + "=" * 60)
    print("Saving FullAttention PDF Heatmaps")
    print("=" * 60)
    save_fullattention_layer_pdfs(attentions, output_dir, seq_len)
    save_fullattention_head_pdfs(attentions, output_dir, seq_len)
    print("\n" + "=" * 60)
    print("PDF Export Complete!")
    print("=" * 60)


def main(args):
    """主函数"""
    print("=" * 60)
    print("FullAttention Query-Key Attention Visualization")
    print("=" * 60)

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print("\n[1/4] Loading model and data...")
    try:
        model, test_loader, args = load_model_and_data(args)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 提取注意力
    print("\n[2/4] Extracting attention weights...")
    attentions = extract_attention_weights(model, test_loader, device, args, max_batches=args.max_batches)

    if not attentions:
        print("No attention weights extracted!")
        return

    # 获取序列长度
    sample_batch = attentions[0]['input']
    seq_len = min(sample_batch.shape[1], args.seq_len)

    # 绘制热力图
    print("\n[3/4] Generating attention heatmaps...")
    attn_data = attentions[0]['attentions']

    # 基本热力图
    save_path = os.path.join(output_dir, 'attention_heatmap.png')
    visualize_full_attention_heatmap(
        attn_data,
        seq_len,
        save_path,
        title="FullAttention: Query-Key Attention Weights"
    )

    # 多头对比
    save_path = os.path.join(output_dir, 'multi_head_attention.png')
    visualize_full_multi_head(attentions, seq_len, save_path)

    # 高级分析
    save_path = os.path.join(output_dir, 'attention_analysis.png')
    visualize_full_attention_advanced(attentions, seq_len, save_path)

    # 保存 PDF
    print("\n[4/4] Saving PDF heatmaps...")
    save_fullattention_pdfs(attentions, output_dir, seq_len)

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FullAttention Visualization')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')

    parser.add_argument('--project_root', type=str, default='.',
                        help='Project root directory')

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
                        help='Label length')

    parser.add_argument('--pred_len', type=int, default=720,
                        help='Prediction sequence length')

    parser.add_argument('--scale', type=bool, default=True,
                        help='Whether to scale the data')

    parser.add_argument('--inverse', type=bool, default=False,
                        help='Whether to inverse the data')

    parser.add_argument('--columns', type=list, default=None,
                        help='Columns to use')

    parser.add_argument('--embed', type=str, default='timeF',
                        help='Time features encoding method')

    parser.add_argument('--freq', type=str, default='h',
                        help='Frequency for time features')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loader')

    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')

    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='Use GPU')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index')

    parser.add_argument('--use_multi_gpu', type=bool, default=False,
                        help='Use multiple GPUs')

    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='Device indices')

    parser.add_argument('--output_dir', type=str, default='./attention_results',
                        help='Output directory')

    parser.add_argument('--max_batches', type=int, default=5,
                        help='Maximum number of batches to process')

    args = parser.parse_args()

    # 设置设备
    if args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.use_gpu = torch.cuda.is_available() and args.use_gpu

    print("\n" + "=" * 60)
    print("Parsed Arguments:")
    print(f"  model_path: {args.model_path}")
    print(f"  project_root: {args.project_root}")
    print(f"  seq_len: {args.seq_len}")
    print(f"  pred_len: {args.pred_len}")
    print(f"  output_dir: {args.output_dir}")
    print("=" * 60)

    main(args)

