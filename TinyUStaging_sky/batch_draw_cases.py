import os
import sys
import numpy as np

# =========================
# 项目根目录
# =========================
project_root = "/export/SHELAB/sikaiyue/sleep/TinyUStaging_sky"
sys.path.insert(0, project_root)

# 复用 evaluate.py 里的绘图函数
from ustaging.bin.evaluate import plot_hypnogram, plot_cm

# =========================
# 输入输出路径
# =========================
base_dir = "/export/SHELAB/sikaiyue/sleep/TinyUStaging_sky/demo/usleeptype1/prefold0/test_data/dcsm"
pred_dir = os.path.join(base_dir, "majority")
out_dir = "/export/SHELAB/sikaiyue/sleep/TinyUStaging_sky/demo/usleeptype1/prefold0_singleplots"

# 创建输出目录
os.makedirs(os.path.join(out_dir, "plots", "hypnograms"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "plots", "CMs"), exist_ok=True)

# =========================
# 找到所有 TRUE 文件
# =========================
true_files = sorted([
    f for f in os.listdir(base_dir)
    if f.endswith("_TRUE.npy")
])

print(f"找到 {len(true_files)} 个样本。")

success = 0
failed = []

for idx, true_file in enumerate(true_files, 1):
    sample_id = true_file.replace("_TRUE.npy", "")
    true_path = os.path.join(base_dir, true_file)
    pred_path = os.path.join(pred_dir, f"{sample_id}_PRED.npy")

    print(f"[{idx}/{len(true_files)}] 处理样本: {sample_id}")

    if not os.path.exists(pred_path):
        print(f"  [跳过] 找不到预测文件: {pred_path}")
        failed.append((sample_id, "missing_pred"))
        continue

    try:
        y_true = np.load(true_path)
        y_pred = np.load(pred_path)

        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)

        if len(y_true) != len(y_pred):
            print(f"  [跳过] 长度不一致: true={len(y_true)}, pred={len(y_pred)}")
            failed.append((sample_id, "length_mismatch"))
            continue

        # plot_cm 里 true 一般按 (N,1) 更稳
        if y_true.ndim == 1:
            y_true_cm = y_true.reshape(-1, 1)
        else:
            y_true_cm = y_true

        # 绘制单个睡眠图
        plot_hypnogram(
            out_dir=out_dir,
            pred=y_pred,
            id_=sample_id,
            true=y_true_cm
        )

        # 绘制单个混淆矩阵
        plot_cm(
            out_dir=out_dir,
            pred=y_pred,
            true=y_true_cm,
            n_classes=5,
            id_=sample_id,
            ignore_classes=5
        )

        success += 1
        print("  [完成]")

    except Exception as e:
        print(f"  [失败] {e}")
        failed.append((sample_id, str(e)))

print("\n========== 批量绘制完成 ==========")
print(f"成功: {success}")
print(f"失败: {len(failed)}")

if failed:
    print("\n失败样本列表：")
    for item in failed:
        print(item)

print("\nHypnogram 输出目录:")
print(os.path.join(out_dir, "plots", "hypnograms"))

print("\nCM 输出目录:")
print(os.path.join(out_dir, "plots", "CMs"))

