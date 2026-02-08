import os
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_item(x):
    # np.load(...).item() works for 0-d object arrays
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()
    return x


def _load_original(here: str):
    label_path = os.path.join(here, 'Y_positive.npy')
    fc_path = os.path.join(here, 'FC.npy')

    raw_label = np.load(label_path, allow_pickle=True)
    raw_fc = np.load(fc_path, allow_pickle=True)

    raw_label = np.asarray(raw_label).reshape(-1)
    raw_fc = np.asarray(raw_fc)

    return label_path, fc_path, raw_label, raw_fc


def _load_merged_dict(path: str):
    obj = np.load(path, allow_pickle=True)
    obj = _safe_item(obj)
    if not isinstance(obj, dict):
        raise ValueError(f'{path} 不是 dict（期望类似 {{"corr":..., "label":...}}），实际类型：{type(obj)}')
    if 'corr' not in obj or 'label' not in obj:
        raise KeyError(f'{path} 缺少 corr/label 键：keys={list(obj.keys())}')
    corr = np.asarray(obj['corr'])
    label = np.asarray(obj['label']).reshape(-1)
    return label, corr


def _label_summary(label: np.ndarray):
    uniq, cnt = np.unique(label, return_counts=True)
    return uniq, cnt


def _plot_label_counts(label: np.ndarray, title: str, out_path: str):
    uniq, cnt = _label_summary(label)

    plt.figure(figsize=(10, 4))
    plt.bar([str(u) for u in uniq], cnt)
    plt.title(title)
    plt.xlabel('label value')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_heatmap(mat: np.ndarray, title: str, out_path: str, vmin=None, vmax=None):
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_hist(values: np.ndarray, title: str, out_path: str, bins: int = 200):
    values = values[np.isfinite(values)]
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel('value')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _stats_corr(name: str, corr: np.ndarray):
    finite = np.isfinite(corr)
    nan_count = np.isnan(corr).sum()
    inf_count = np.isinf(corr).sum()
    total = corr.size

    # sample symmetry check on first few graphs (cheap)
    symmetry_diffs = []
    for i in range(min(3, corr.shape[0])):
        a = corr[i]
        if a.ndim == 2 and a.shape[0] == a.shape[1]:
            symmetry_diffs.append(np.nanmax(np.abs(a - a.T)))
    sym = float(np.max(symmetry_diffs)) if symmetry_diffs else float('nan')

    return {
        'name': name,
        'shape': tuple(corr.shape),
        'dtype': str(corr.dtype),
        'nan': int(nan_count),
        'inf': int(inf_count),
        'finite_pct': float(finite.sum() * 100.0 / total),
        'min': float(np.nanmin(corr)),
        'max': float(np.nanmax(corr)),
        'mean': float(np.nanmean(corr)),
        'symmetry_max_abs_diff_first3': sym,
    }


def main():
    here = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(here, 'viz_out')
    _ensure_dir(out_dir)

    # 1) 原始
    label_path, fc_path, raw_label, raw_fc = _load_original(here)

    # 2) 合并后
    merged_path = os.path.join(here, 'FC_Y.npy')
    merged_label, merged_fc = _load_merged_dict(merged_path)

    # 3) 可选：ABIDE 格式
    abide_path = os.path.join(here, 'ABIDE', 'abide.npy')
    abide_label, abide_fc = (None, None)
    if os.path.exists(abide_path):
        abide_label, abide_fc = _load_merged_dict(abide_path)

    # --- 数值核对：corr 是否被改动 ---
    if raw_fc.shape != merged_fc.shape:
        raise ValueError(f'corr shape 不一致：raw={raw_fc.shape} merged={merged_fc.shape}')
    diff = np.asarray(raw_fc) - np.asarray(merged_fc)
    max_abs_diff = float(np.nanmax(np.abs(diff)))

    # --- 打印摘要 ---
    print('=== Paths ===')
    print('raw label :', label_path)
    print('raw corr  :', fc_path)
    print('merged    :', merged_path)
    if abide_label is not None:
        print('abide     :', abide_path)
    print()

    print('=== Shapes ===')
    print('raw label shape :', raw_label.shape, 'dtype:', raw_label.dtype)
    print('raw corr  shape :', raw_fc.shape, 'dtype:', raw_fc.dtype)
    print('merged label shape:', merged_label.shape, 'dtype:', merged_label.dtype)
    print('merged corr  shape:', merged_fc.shape, 'dtype:', merged_fc.dtype)
    if abide_label is not None:
        print('abide label shape :', abide_label.shape, 'dtype:', abide_label.dtype)
        print('abide corr  shape :', abide_fc.shape, 'dtype:', abide_fc.dtype)
    print()

    print('=== Label Summary ===')
    u0, c0 = _label_summary(raw_label)
    u1, c1 = _label_summary(merged_label)
    print('raw unique labels   :', u0)
    print('raw label counts    :', c0)
    print('merged unique labels:', u1)
    print('merged label counts :', c1)
    print()

    print('=== Corr Consistency ===')
    print('max_abs_diff(raw_corr - merged_corr) =', max_abs_diff)
    print()

    print('=== Corr Stats ===')
    print(_stats_corr('raw_corr', raw_fc))
    print(_stats_corr('merged_corr', merged_fc))
    if abide_label is not None:
        print(_stats_corr('abide_corr', abide_fc))
    print()

    # --- 画图 ---
    _plot_label_counts(raw_label, 'Raw labels (Y_positive.npy)', os.path.join(out_dir, 'raw_label_counts.png'))
    _plot_label_counts(merged_label, 'Merged labels (FC_Y.npy)', os.path.join(out_dir, 'merged_label_counts.png'))

    # heatmap：第 0 个样本
    i = 0
    _plot_heatmap(raw_fc[i], f'Raw corr heatmap (sample {i})', os.path.join(out_dir, f'raw_corr_sample_{i}.png'))
    _plot_heatmap(merged_fc[i], f'Merged corr heatmap (sample {i})', os.path.join(out_dir, f'merged_corr_sample_{i}.png'))

    # 均值矩阵
    raw_mean = np.nanmean(raw_fc, axis=0)
    merged_mean = np.nanmean(merged_fc, axis=0)
    vmin = float(min(np.nanmin(raw_mean), np.nanmin(merged_mean)))
    vmax = float(max(np.nanmax(raw_mean), np.nanmax(merged_mean)))
    _plot_heatmap(raw_mean, 'Raw corr mean heatmap', os.path.join(out_dir, 'raw_corr_mean.png'), vmin=vmin, vmax=vmax)
    _plot_heatmap(merged_mean, 'Merged corr mean heatmap', os.path.join(out_dir, 'merged_corr_mean.png'), vmin=vmin, vmax=vmax)

    # 值分布（随机采样 200k，避免直方图过慢）
    rng = np.random.default_rng(0)
    flat = raw_fc.reshape(-1)
    n = min(200_000, flat.size)
    idx = rng.choice(flat.size, size=n, replace=False)
    _plot_hist(flat[idx], 'Raw corr value histogram (sampled)', os.path.join(out_dir, 'raw_corr_hist.png'))

    flat2 = merged_fc.reshape(-1)
    n2 = min(200_000, flat2.size)
    idx2 = rng.choice(flat2.size, size=n2, replace=False)
    _plot_hist(flat2[idx2], 'Merged corr value histogram (sampled)', os.path.join(out_dir, 'merged_corr_hist.png'))

    if abide_label is not None:
        _plot_label_counts(abide_label, 'ABIDE labels (ABIDE/abide.npy)', os.path.join(out_dir, 'abide_label_counts.png'))
        _plot_heatmap(abide_fc[0], 'ABIDE corr heatmap (sample 0)', os.path.join(out_dir, 'abide_corr_sample_0.png'))

    print('=== Saved Plots ===')
    print(out_dir)


if __name__ == '__main__':
    main()
