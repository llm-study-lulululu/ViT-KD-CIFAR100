import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm # 一个优雅的进度条库
import os

# --- 1. 项目配置与超参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64         # 3090使用64或128通常没问题
EPOCHS = 25             # 完整的训练轮数
LEARNING_RATE = 3e-5    # ViT微调常用的学习率
NUM_LABELS = 100        # CIFAR-100数据集有100个类别

# 知识蒸馏 (KD) 相关的超参数
# 通过修改ALPHA的值来切换实验模式
ALPHA = 0.7
TEMPERATURE = 2         # 蒸馏温度，用于平滑教师模型的输出

# --- 2. 数据加载与预处理模块 ---
def get_dataloaders(batch_size=64):
    """加载并预处理CIFAR-100数据集，返回训练和测试的DataLoader"""
    print("--- 正在加载和预处理数据... ---")
    # --- 新增：定义一个永久的缓存目录 ---
    CACHE_PATH = os.path.expanduser('~/huggingface_cache')
    print(f"--- 数据集将缓存到: {CACHE_PATH} ---")
    NUM_WORKERS = 20 
    # 定义图像预处理流程
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # 将图像放大以匹配ViT的输入尺寸
        transforms.ToTensor(), # 将PIL图像转换为PyTorch张量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # 归一化
    ])

    def preprocess_images(examples):
        # 对Hugging Face datasets中的每个图像应用上面的变换
        examples['pixel_values'] = [image_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    # 加载完整的CIFAR-100数据集
    train_dataset = load_dataset('cifar100', split='train', cache_dir=CACHE_PATH)
    test_dataset = load_dataset('cifar100', split='test', cache_dir=CACHE_PATH)

    # 使用.map()方法高效地对整个数据集进行预处理
    train_dataset = train_dataset.map(preprocess_images, batched=True, remove_columns=['img', 'coarse_label'])
    test_dataset = test_dataset.map(preprocess_images, batched=True, remove_columns=['img', 'coarse_label'])
    
    # 重命名标签列以匹配模型期望的输入名称'labels'
    train_dataset = train_dataset.rename_column("fine_label", "labels")
    test_dataset = test_dataset.rename_column("fine_label", "labels")

    # 设置数据集格式为PyTorch张量
    train_dataset.set_format('torch')
    test_dataset.set_format('torch')

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=NUM_WORKERS)
    
    print("--- 数据准备完成! ---")
    return train_loader, test_loader

# --- 3. 模型加载模块 ---
def get_models():
    """加载教师和学生模型"""
    print("--- 正在加载模型... ---")
    
    CACHE_PATH = os.path.expanduser('~/huggingface_cache')
    FINETUNED_TEACHER_PATH = "vit_base_cifar100_finetuned.pth"

    # --- 加载教师模型 ---
    teacher_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True, # 
        cache_dir=CACHE_PATH
    )
    
    
    try:
        print(f"--- 正在从 {FINETUNED_TEACHER_PATH} 加载微调后的教师权重... ---")
        teacher_model.load_state_dict(torch.load(FINETUNED_TEACHER_PATH))
        print("--- ✅ 教师权重加载成功! ---")
    except FileNotFoundError:
        print(f"--- ⚠️ 警告: 未找到微调后的教师权重。将使用Hugging Face的预训练权重进行蒸馏。 ---")
    
    teacher_model.to(DEVICE)

    # --- 加载学生模型 ---
    student_model = ViTForImageClassification.from_pretrained(
        'facebook/deit-small-patch16-224', 
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
        cache_dir=CACHE_PATH 
    ).to(DEVICE)
    
    print("--- 模型加载完成! ---")
    return teacher_model, student_model
# --- 4. 训练与评估逻辑 ---
def train_one_epoch(teacher, student, loader, optimizer, alpha, temp):
    """执行一个epoch的训练"""
    teacher.eval() # 教师模型进入评估模式，不进行训练
    student.train() # 学生模型进入训练模式

    total_loss = 0.0
    for batch in tqdm(loader, desc="训练中"):
        # 将数据移动到指定的设备 (GPU)
        pixel_values = batch['pixel_values'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        # 获取教师和学生的输出 (logits)
        with torch.no_grad(): # 关闭教师模型的梯度计算
            teacher_outputs = teacher(pixel_values).logits
        
        student_outputs = student(pixel_values).logits

        # --- 计算损失函数 L_total = α * L_soft + (1 - α) * L_hard ---
        # L_hard: 学生与真实标签的标准交叉熵损失
        loss_hard = nn.CrossEntropyLoss()(student_outputs, labels)

        # L_soft: 学生与教师软标签的KL散度损失
        # KLDivLoss期望输入是log-probabilities，所以对学生输出用log_softmax
        loss_soft = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_outputs / temp, dim=1),
            F.softmax(teacher_outputs / temp, dim=1) # 教师输出用softmax
        ) * (temp * temp) # 乘以温度的平方是Hinton论文中的一个技巧，用于保持梯度尺度

        # 计算总损失
        loss = alpha * loss_soft + (1 - alpha) * loss_hard
        
        # --- 反向传播与优化 ---
        optimizer.zero_grad() # 清空之前的梯度
        loss.backward() # 计算梯度
        optimizer.step() # 更新学生模型的权重
        
        total_loss += loss.item() #累加loss
        
    return total_loss / len(loader) # 返回平均loss

def evaluate(model, loader):
    """在测试集上评估模型性能"""
    model.eval() # 模型进入评估模式
    correct = 0
    total = 0
    with torch.no_grad(): # 关闭梯度计算
        for batch in tqdm(loader, desc="评估中"):
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(pixel_values).logits
            _, predicted = torch.max(outputs.data, 1) # 获取概率最高的类别作为预测结果
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total # 返回准确率

# --- 5. 主执行函数 ---
if __name__ == "__main__":
    # 1. 准备数据和模型
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    teacher_model, student_model = get_models()

    # 2. 冻结教师模型的参数，我们不训练它
    for param in teacher_model.parameters():
        param.requires_grad = False
        
    # 3. 创建优化器，只传入学生模型的参数，确保只更新学生
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    
    # 4. 打印实验配置信息
    print("\n--- 开始训练 ---")
    print(f"实验模式: {'知识蒸馏 (KD)' if ALPHA > 0 else '独立训练 (Baseline)'}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}")
    if ALPHA > 0:
        print(f"Alpha: {ALPHA}, Temperature: {TEMPERATURE}")
    
    # 5. 执行训练和评估循环
    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(teacher_model, student_model, train_loader, optimizer, ALPHA, TEMPERATURE)
        accuracy = evaluate(student_model, test_loader)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  平均训练损失 (Average Train Loss): {avg_loss:.4f}")
        print(f"  测试集准确率 (Test Accuracy): {accuracy:.2f}%")
        
    print("\n--- 训练完成! ---")