#!/usr/bin/env python
"""
统计睡眠分期dcsm文件中各阶段的epoch数（每30秒一个epoch）



使用方法：
# 统计单个文件
python count_stages.py /path/to/hypnogram.ids

# 统计一个折中的所有test文件
find /export/SHELAB/sikaiyue/sleep/TinyUStaging_sky/demo/datasets/processed_SKY/dcsm/views/10_CV/split_0/test -name "*.ids" | xargs python count_stages.py

# 统计所有折
for i in {0..9}; do
    echo "========== Fold $i =========="
    find /export/SHELAB/sikaiyue/sleep/TinyUStaging_sky/demo/datasets/processed_SKY/dcsm/views/10_CV/split_${i}/test -name "*.ids" | xargs python count_stages.py
done
"""
import sys
from collections import Counter, defaultdict

def count_epochs_from_ids(file_path):
    """
    从.ids文件统计各睡眠阶段的epoch数量
    
    Args:
        file_path: .ids文件路径
    
    Returns:
        dict: 各阶段对应的epoch数量
    """
    stage_counts = defaultdict(int)
    total_epochs = 0
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            
            try:
                # 解析每一行
                start, duration, stage = line.split(',')
                start = int(start)
                duration = int(duration)
                
                # 计算这段持续时间包含多少个epoch（每30秒一个）
                n_epochs = duration // 30
                stage_counts[stage] += n_epochs
                total_epochs += n_epochs
                
                # 可选：检查是否有不是30秒整数倍的情况
                if duration % 30 != 0:
                    print(f"警告: 第{line_num}行持续时间{duration}秒不是30的整数倍")
                    
            except Exception as e:
                print(f"解析第{line_num}行出错: {line}")
                print(f"错误信息: {e}")
    
    return dict(stage_counts), total_epochs

def main():
    if len(sys.argv) < 2:
        print("用法: python count_stages.py <ids_file1> <ids_file2> ...")
        print("或: python count_stages.py 文件名")
        sys.exit(1)
    
    # 可以统计多个文件
    all_counts = Counter()
    all_total = 0
    
    for file_path in sys.argv[1:]:
        print(f"\n=== 统计文件: {file_path} ===")
        counts, total = count_epochs_from_ids(file_path)
        
        # 打印这个文件的统计结果
        print(f"总epoch数: {total}")
        for stage in sorted(counts.keys()):
            percentage = counts[stage] / total * 100
            print(f"  {stage}: {counts[stage]} ({percentage:.2f}%)")
        
        all_counts.update(counts)
        all_total += total
    
    # 打印总计
    if len(sys.argv) > 2:
        print("\n" + "="*50)
        print("所有文件总计:")
        print(f"总epoch数: {all_total}")
        for stage in sorted(all_counts.keys()):
            percentage = all_counts[stage] / all_total * 100
            print(f"  {stage}: {all_counts[stage]} ({percentage:.2f}%)")

if __name__ == "__main__":
    main()