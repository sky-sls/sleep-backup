#!/usr/bin/env python
"""
统计10折交叉验证中每折test集的睡眠分期分布，并输出到文件
"""

import sys
import os
from collections import Counter, defaultdict
from datetime import datetime

def count_epochs_from_ids(file_path):
    """
    从.ids文件统计各睡眠阶段的epoch数量
    """
    stage_counts = defaultdict(int)
    total_epochs = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                start, duration, stage = line.split(',')
                duration = int(duration)
                n_epochs = duration // 30
                stage_counts[stage] += n_epochs
                total_epochs += n_epochs
            except:
                continue
    
    return dict(stage_counts), total_epochs

def main():
    # 输出文件路径
    output_file = "dcsm_stages_statistics.txt"
    
    # 打开输出文件
    with open(output_file, 'w') as f_out:
        # 写入头部
        f_out.write("=" * 60 + "\n")
        f_out.write("DCSM数据集10折交叉验证test集统计\n")
        f_out.write(f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f_out.write("=" * 60 + "\n\n")
        
        # 存储所有折的总计
        all_folds_total = Counter()
        all_folds_files = set()
        
        # 统计每一折
        for fold in range(10):
            f_out.write(f"\n{'='*50}\n")
            f_out.write(f"Fold {fold} 统计\n")
            f_out.write(f"{'='*50}\n")
            
            # 构建路径
            test_path = f"/export/SHELAB/sikaiyue/sleep/TinyUStaging_sky/demo/datasets/processed_SKY/dcsm/views/10_CV/split_{fold}/test"
            
            # 使用find命令获取所有.ids文件（跟随软链接）
            import subprocess
            cmd = f"find -L {test_path} -name '*.ids' -type f"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            files = result.stdout.strip().split('\n')
            
            if not files or files[0] == '':
                f_out.write(f"警告: Fold {fold} 没有找到.ids文件\n")
                continue
            
            # 统计这一折
            fold_counts = Counter()
            fold_total = 0
            file_count = 0
            
            for file_path in files:
                if not file_path:
                    continue
                counts, total = count_epochs_from_ids(file_path)
                fold_counts.update(counts)
                fold_total += total
                file_count += 1
                all_folds_files.add(file_path)
            
            # 写入这一折的统计结果
            f_out.write(f"文件数: {file_count}\n")
            f_out.write(f"总epoch数: {fold_total}\n")
            f_out.write("各阶段分布:\n")
            
            for stage in sorted(fold_counts.keys()):
                count = fold_counts[stage]
                percentage = (count / fold_total) * 100 if fold_total > 0 else 0
                f_out.write(f"  {stage}: {count:6d} ({percentage:5.2f}%)\n")
            
            # 累加到总计
            all_folds_total.update(fold_counts)
        
        # 写入所有折的总计
        f_out.write("\n" + "=" * 60 + "\n")
        f_out.write("所有10折test集总计 (原始数据集)\n")
        f_out.write("=" * 60 + "\n")
        
        total_files = len(all_folds_files)
        total_epochs_all = sum(all_folds_total.values())
        
        f_out.write(f"\n总文件数: {total_files} (应该是255个)\n")
        f_out.write(f"总epoch数: {total_epochs_all}\n")
        f_out.write("\n各阶段总计:\n")
        
        for stage in sorted(all_folds_total.keys()):
            count = all_folds_total[stage]
            percentage = (count / total_epochs_all) * 100
            f_out.write(f"  {stage}: {count:8d} ({percentage:6.2f}%)\n")
        
        f_out.write("\n" + "=" * 60 + "\n")
        f_out.write("统计完成！\n")
    
    print(f"统计结果已保存到: {output_file}")
    print(f"总文件数: {total_files}")

if __name__ == "__main__":
    main()
