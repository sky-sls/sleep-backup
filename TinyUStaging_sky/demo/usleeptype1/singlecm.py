import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ===== 字体设置（关键）=====
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']   # 英文
matplotlib.rcParams['font.sans-serif'] = ['SimSun']       # 中文
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== 类别标签 =====
labels = ["W","REM","N1","N2","N3"]

# ===== 四个归一化矩阵 =====
matrices = {
    "ECG": np.array([
        [0.845,0.062,0.030,0.044,0.019],
        [0.108,0.743,0.028,0.083,0.044],
        [0.256,0.163,0.404,0.237,0.138],
        [0.061,0.039,0.055,0.794,0.149],
        [0.053,0.062,0.032,0.201,0.653]
    ]),
    "ABD": np.array([
        [0.809,0.070,0.035,0.050,0.025],
        [0.131,0.704,0.031,0.091,0.043],
        [0.281,0.170,0.336,0.264,0.140],
        [0.068,0.045,0.063,0.748,0.173],
        [0.062,0.070,0.035,0.218,0.615]
    ]),
    "THO": np.array([
        [0.801,0.072,0.036,0.052,0.027],
        [0.136,0.688,0.033,0.093,0.050],
        [0.295,0.184,0.313,0.270,0.138],
        [0.072,0.048,0.066,0.736,0.175],
        [0.065,0.076,0.037,0.227,0.595]
    ]),
    "NASAL": np.array([
        [0.788,0.077,0.038,0.056,0.027],
        [0.143,0.662,0.036,0.100,0.060],
        [0.312,0.199,0.276,0.274,0.139],
        [0.078,0.052,0.069,0.710,0.187],
        [0.072,0.083,0.043,0.235,0.567]
    ])
}

# ===== 绘图函数 =====
def plot_cm(cm, title, save_path):
    plt.figure(figsize=(6,5))
    plt.imshow(cm)

    plt.xticks(range(5), labels, fontsize=11)
    plt.yticks(range(5), labels, fontsize=11)

    for i in range(5):
        for j in range(5):
            plt.text(j, i, f"{cm[i,j]:.3f}",
                     ha="center", va="center", fontsize=10)

    plt.xlabel("Predicted label", fontsize=12)
    plt.ylabel("True label", fontsize=12)
    plt.title(title, fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

# ===== 批量生成 =====
for name, cm in matrices.items():
    plot_cm(cm, f"{name} Confusion Matrix", f"{name}_cm.pdf")

print("全部生成完成！")