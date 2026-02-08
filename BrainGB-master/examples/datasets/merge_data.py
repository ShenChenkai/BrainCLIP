import os
import numpy as np


def main():
	here = os.path.dirname(os.path.realpath(__file__))
	label_path = os.path.join(here, 'Y_positive.npy')
	corr_path = os.path.join(here, 'FC.npy')
	out_dir = os.path.join(here, 'ABIDE')
	out_path = os.path.join(out_dir, 'abide.npy')
	out_custom_path = os.path.join(here, 'FC_Y.npy')

	os.makedirs(out_dir, exist_ok=True)

	label = np.load(label_path, allow_pickle=True)
	corr = np.load(corr_path, allow_pickle=True)

	label = np.asarray(label).reshape(-1)
	corr = np.asarray(corr)

	# BrainGB 的 examples 训练/评估逻辑默认是二分类：
	# - loss: F.nll_loss(out, y)
	# - AUC: 使用第 2 类概率 torch.exp(out)[:, 1]
	# 因此这里将 label 规范为 {0,1}。
	uniq = np.unique(label)
	if not (len(uniq) <= 2 and set(uniq.tolist()).issubset({0, 1})):
		# 默认：把 0 视为负类，非 0 视为正类
		label = (label != 0).astype(np.int64)

	if corr.ndim != 3:
		raise ValueError(f'FC.npy 需要是 3 维数组 (#sub, #ROI, #ROI)，但当前是 {corr.shape}')
	if corr.shape[0] != label.shape[0]:
		raise ValueError(f'样本数不一致：label={label.shape[0]}，corr={corr.shape[0]}')
	if corr.shape[1] != corr.shape[2]:
		raise ValueError(f'每个样本的相关矩阵必须是方阵，但当前是 {corr.shape[1:]}')

	payload = {
		# BrainGB 的 ABIDE loader 期望键名为 "corr" 和 "label"（小写）
		'corr': corr,
		'label': label,
	}
	np.save(out_path, payload)
	np.save(out_custom_path, payload)

	print('合并完成：')
	print(f'- label: {label_path} -> shape={label.shape}, dtype={label.dtype}')
	print(f'- corr : {corr_path} -> shape={corr.shape}, dtype={corr.dtype}')
	print(f'- output -> {out_path}')
	print(f'- output -> {out_custom_path}')


if __name__ == '__main__':
	main()
