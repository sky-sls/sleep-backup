import os
import numpy as np
from collections import Counter

base_dir = "/export/SHELAB/sikaiyue/sleep/TinyUStaging_sky/demo/usleeptype1/valfolds_best_fold_0/val_data/dcsm/files"

true_counter = Counter()
pred_counter = Counter()
subject_num = 0

for subject in os.listdir(base_dir):
    subject_dir = os.path.join(base_dir, subject)
    if not os.path.isdir(subject_dir):
        continue

    true_path = os.path.join(subject_dir, "true.npz")
    pred_path = os.path.join(subject_dir, "pred.npz")

    if not (os.path.exists(true_path) and os.path.exists(pred_path)):
        print(f"跳过 {subject}，缺少 true.npz 或 pred.npz")
        continue

    true_data = np.load(true_path)
    pred_data = np.load(pred_path)

    # 查看 key
    true_keys = list(true_data.keys())
    pred_keys = list(pred_data.keys())

    # 自动取第一个数组
    y_true = true_data[true_keys[0]]
    y_pred = pred_data[pred_keys[0]]

    # 展平
    y_true = y_true.reshape(-1)

    # pred 可能已经是标签，也可能是概率
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    y_pred = y_pred.reshape(-1)

    true_counter.update(y_true.tolist())
    pred_counter.update(y_pred.tolist())

    subject_num += 1

print(f"成功统计个体数: {subject_num}")

print("\n真实标签五分类统计:")
for i in range(5):
    print(f"class {i}: {true_counter[i]}")

print("\n预测标签五分类统计:")
for i in range(5):
    print(f"class {i}: {pred_counter[i]}")