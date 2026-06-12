#!/usr/bin/env python
"""
测试 DafmSleepNet 模型是否可以正常初始化和前向传播
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ustaging.models import DafmSleepNet
from mpunet.logging import ScreenLogger

print("=" * 60)
print("测试 DafmSleepNet 模型")
print("=" * 60)

# 创建 logger
logger = ScreenLogger()

# 模型超参数（根据 pre_proc_hparams.yaml）
hparams = {
    'n_classes': 5,
    'batch_shape': [32, 11, 3840, 2],
    'depth': 4,
    'dilation': ((1, 2), (5, 1), (2, 5), (1, 2)),
    'activation': 'elu',
    'dense_classifier_activation': 'tanh',
    'kernel_size': 5,
    'transition_window': 5,
    'padding': 'same',
    'init_filters': 16,
    'complexity_factor': 2,
    'l2_reg': None,
    'pools': (10, 8, 6, 4),
    'data_per_prediction': None,
    'attention_module_encoder': 'SE',
    'isChanAttn': True,
    'isSeparable': False,
    'attention_module_encoder_layer': 3,
    'attention_module_bottom': 'SE',
    'attention_module_dense': 'SE',
    'attention_module_seq': 'SE',
    'ratio_se': [1, 8, 4, 1],
    'ratio_cbam': None,
    'isAdd': False,
    'logger': logger,
    'build': True,
    # 域对抗学习参数（先测试关闭状态）
    'use_domain_adversarial': False,
    'n_domains': 10,
    'lambda_val': 1.0,
    'domain_hidden_units': 256
}

print("\n1. 测试不启用域对抗学习的模型...")
print("-" * 60)

try:
    model = DafmSleepNet(**hparams)
    print("✓ 模型初始化成功！")
    print(f"  - 模型输入: {model.input}")
    print(f"  - 模型输出: {model.output}")
    
    # 创建测试数据
    test_input = np.random.randn(*hparams['batch_shape']).astype(np.float32)
    print(f"\n  测试输入形状: {test_input.shape}")
    
    # 前向传播
    test_output = model.predict(test_input, verbose=1)
    print(f"  测试输出形状: {test_output.shape}")
    print("✓ 前向传播成功！")
    
except Exception as e:
    print(f"✗ 错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("2. 测试启用域对抗学习的模型...")
print("-" * 60)

hparams_with_domain = hparams.copy()
hparams_with_domain['use_domain_adversarial'] = True

try:
    model_with_domain = DafmSleepNet(**hparams_with_domain)
    print("✓ 启用域对抗的模型初始化成功！")
    print(f"  - 模型输入: {model_with_domain.input}")
    print(f"  - 模型输出数量: {len(model_with_domain.output)}")
    for i, out in enumerate(model_with_domain.output):
        print(f"  - 输出 {i+1} 形状: {out.shape}")
    
    # 创建测试数据
    test_input = np.random.randn(*hparams['batch_shape']).astype(np.float32)
    print(f"\n  测试输入形状: {test_input.shape}")
    
    # 前向传播
    test_outputs = model_with_domain.predict(test_input, verbose=1)
    print(f"  睡眠分期输出形状: {test_outputs[0].shape}")
    print(f"  域分类输出形状: {test_outputs[1].shape}")
    print("✓ 域对抗模型前向传播成功！")
    
except Exception as e:
    print(f"✗ 错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
