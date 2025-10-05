import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

print("--- 最终成果汇报绘图脚本开始运行 ---")
# 1. 定义解析函数
def parse_log_for_accuracy(log_file):
    """读取指定的log文件，抽取出每一轮的测试集准确率"""
    accuracies = []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '测试集准确率' in line or 'Test Accuracy' in line:
                    match = re.search(r'(\d+\.\d+)', line)
                    if match:
                        accuracies.append(float(match.group(1)))
    except FileNotFoundError:
        print(f"!!! 警告: 未找到日志文件 {log_file}。将跳过此文件。")
    return accuracies
# 2. 准备所有5个实验的数据
print("--- 正在解析所有日志文件... ---")
log_files = {
    'Teacher': 'finetune_teacher.log',
    'Baseline Student': 'baseline_training.log',
    'Distilled (α=0.7, T=4)': 'kd_alpha_0.7_temp_4.log',
    'Distilled (α=0.5, T=4)': 'kd_alpha_0.5_temp_4.log',
    'Distilled (α=0.7, T=2)': 'kd_alpha_0.7_temp_2.log' 
}

all_accuracies = {name: parse_log_for_accuracy(file) for name, file in log_files.items()}

# 检查数据是否都有效
if any(not acc_list for acc_list in all_accuracies.values()):
    print("\n!!! 严重错误: 未能从一个或多个日志文件中成功提取数据。无法生成图表。")
else:
    # --- 准备最终准确率数据 ---
    final_accuracy_data = {
        'Model Type': [
            f'Teacher\n(ViT-Base, ~86M)', 
            f'Baseline\n(ViT-Small, ~22M)', 
            f'Distilled (α=0.7, T=4)\n(ViT-Small, ~22M)',
            f'Distilled (α=0.5, T=4)\n(ViT-Small, ~22M)',
            f'Distilled (α=0.7, T=2)\n(ViT-Small, ~22M)'
        ],
        'Test Accuracy (%)': [
            all_accuracies['Teacher'][-1],
            all_accuracies['Baseline Student'][-1],
            all_accuracies['Distilled (α=0.7, T=4)'][-1],
            all_accuracies['Distilled (α=0.5, T=4)'][-1],
            all_accuracies['Distilled (α=0.7, T=2)'][-1]
        ]
    }
    df_final_acc = pd.DataFrame(final_accuracy_data)

    # --- 准备训练过程数据 ---
    student_keys = ['Baseline Student', 'Distilled (α=0.7, T=4)', 'Distilled (α=0.5, T=4)', 'Distilled (α=0.7, T=2)']
    df_process_list = []
    for key in student_keys:
        if all_accuracies[key]:
            num_epochs = len(all_accuracies[key])
            epochs = list(range(1, num_epochs + 1))
            df_temp = pd.DataFrame({
                'Epoch': epochs,
                'Test Accuracy (%)': all_accuracies[key],
                'Experiment': key 
            })
            df_process_list.append(df_temp)
    df_process = pd.concat(df_process_list)
    
    
    # 3. 开始绘图
    print("--- 数据准备完毕，开始生成图表... ---")
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.2) 

    # --- 图表一：最终准确率对比柱状图 (完整版) ---
    plt.figure(figsize=(14, 9))
    bar_plot = sns.barplot(x='Model Type', y='Test Accuracy (%)', hue='Model Type', data=df_final_acc, dodge=False)
    
    for p in bar_plot.patches:
        bar_plot.annotate(f"{p.get_height():.2f}%", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12, weight='bold')

    plt.title('Final Model Accuracy vs. Size & Hyperparameters (CIFAR-100)', fontsize=20, weight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=15)
    plt.xlabel('Model Configuration', fontsize=15)
    plt.ylim(84, 92) 
    plt.xticks(rotation=0, ha='center') # 确保标签水平显示
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig('final_results_summary.png', dpi=300)
    print("图表 'final_results_summary.png' 已保存。")

    # --- 图表二：训练过程对比折线图 (完整版) ---
    plt.figure(figsize=(14, 8))
    sns.lineplot(x='Epoch', y='Test Accuracy (%)', hue='Experiment', style='Experiment', markers=True, dashes=False, data=df_process, linewidth=2.5)

    plt.title('All Student Models Training Process (CIFAR-100)', fontsize=20, weight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.legend(title='Experiment', fontsize=12)
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig('training_process_summary.png', dpi=300)
    print("图表 'training_process_summary.png' 已保存。")

    plt.show()
    print("\n--- 脚本执行完毕 ---")