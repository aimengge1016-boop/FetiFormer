import numpy as np
import matplotlib.pyplot as plt


def main():
    # =========================
    # 1. 加载数据
    # =========================
    folder = 'ECL_96_192_iTransformer_custom_M_ft96_sl48_ll192_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0'

    pred = np.load(f'./results/{folder}/pred.npy', mmap_mode='r')
    true = np.load(f'./results/{folder}/true.npy', mmap_mode='r')
    input_x = np.load(f'./results/{folder}/input.npy', mmap_mode='r')

    # 若 input 为 4 维，reshape 成 (N, T, C)
    if input_x.ndim == 4:
        input_x = input_x.reshape(-1, input_x.shape[-2], input_x.shape[-1])

    print('Data loaded:', input_x.shape, pred.shape, true.shape)

    # =========================
    # 2. 选择特征
    # =========================
    feat_idx = -1  # 使用最后一个特征

    # =========================
    # 3. 计算每个样本的预测误差（MAE）
    # =========================
    mae_per_sample = np.mean(
        np.abs(pred[:, :, feat_idx] - true[:, :, feat_idx]),
        axis=1
    )

    best_sample_idx = np.argmin(mae_per_sample)

    print(f'Best sample index: {best_sample_idx}')
    print(f'Best sample MAE: {mae_per_sample[best_sample_idx]:.4f}')

    # =========================
    # 4. 取 best sample 数据
    # =========================
    history = input_x[best_sample_idx, :, feat_idx]
    pred_future = pred[best_sample_idx, :, feat_idx]
    true_future = true[best_sample_idx, :, feat_idx]

    T_in = len(history)
    T_out = len(pred_future)

    # 拼接完整序列
    y_true = np.concatenate([history, true_future])
    y_pred = np.concatenate([history, pred_future])

    x = np.arange(T_in + T_out)
    x_future = np.arange(T_out) + T_in

    # =========================
    # 5. 95% 置信区间（基于 best sample 自身误差）
    # =========================
    sample_error = pred_future - true_future
    sigma = np.std(sample_error)

    ci_upper = pred_future + 1.96 * sigma
    ci_lower = pred_future - 1.96 * sigma

    print(f'CI sigma (best sample): {sigma:.4f}')

    # =========================
    # 6. 绘图（论文级，干净）
    # =========================
    plt.figure(figsize=(12, 4))

    plt.plot(x, y_true, label='Ground Truth', color='#1f77b4')
    plt.plot(x, y_pred, '--', label='Prediction', color='#ff7f0e')

    plt.fill_between(
        x_future,
        ci_lower,
        ci_upper,
        color='#1f77b4',
        alpha=0.15,
        label='95% Confidence Interval'
    )

    plt.axvline(T_in, linestyle='--', color='gray', linewidth=1)

    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('96-Predict-192')
    plt.legend(frameon=False)
    plt.tight_layout()

    # 保存图片
    save_path = './figures/ECL_best_sample_96_192.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f'Figure saved to: {save_path}')

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
