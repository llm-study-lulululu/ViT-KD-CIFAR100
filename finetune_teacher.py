import torch
import torch.nn as nn
from transformers import ViTForImageClassification, AdamW
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- 1. 项目配置与超参数 ---
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 3e-5
NUM_LABELS = 100
MODEL_SAVE_PATH = "vit_base_cifar100_finetuned.pth" 

# --- 2. 数据加载与预处理模块 ---
# ==============================================================================
def get_dataloaders(batch_size=64):
    """加载并预处理CIFAR-100数据集，返回训练和测试的DataLoader"""
    print("--- 正在加载和预处理数据... ---")
    # --- 新增：定义一个永久的缓存目录 ---
    CACHE_PATH = os.path.expanduser('~/huggingface_cache')
    print(f"--- 数据集将缓存到: {CACHE_PATH} ---")
    NUM_WORKERS = 20
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def preprocess_images(examples):
        examples['pixel_values'] = [image_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    train_dataset = load_dataset('cifar100', split='train', cache_dir=CACHE_PATH).map(preprocess_images, batched=True, remove_columns=['img', 'coarse_label'])
    test_dataset = load_dataset('cifar100', split='test', cache_dir=CACHE_PATH).map(preprocess_images, batched=True, remove_columns=['img', 'coarse_label'])
    
    train_dataset = train_dataset.rename_column("fine_label", "labels").with_format('torch')
    test_dataset = test_dataset.rename_column("fine_label", "labels").with_format('torch')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    
    print("--- 数据准备完成! ---")
    return train_loader, test_loader

# --- 3. 模型加载模块 ---
def get_teacher_model():
    print("--- 正在加载 ViT-Base 模型... ---")
    # --- 新增：定义缓存路径 ---
    CACHE_PATH = os.path.expanduser('~/huggingface_cache')
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
        cache_dir=CACHE_PATH
    ).to(DEVICE)
    print("--- 模型加载完成! ---")
    return model

# --- 4. 训练与评估逻辑  ---
def finetune_epoch(model, loader, optimizer):
    model.train() # 模型进入训练模式
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss() # 定义标准交叉熵损失

    for batch in tqdm(loader, desc="微调中"):
        # 将数据移动到GPU
        pixel_values = batch['pixel_values'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 单次前向传播
        outputs = model(pixel_values).logits
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader) # 返回平均损失

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="评估中"):
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(pixel_values).logits
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total

# --- 5. 主执行函数 ---
# ==============================================================================
if __name__ == "__main__":
    # 1. 准备数据和模型
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    # <--- 只加载教师模型
    teacher_model = get_teacher_model()
    
    # 评估微调前的准确率 (随机分类头，准确率应该约等于1/100=1%)
    initial_accuracy = evaluate(teacher_model, test_loader)
    print(f"\nViT-Base 微调前准确率: {initial_accuracy:.2f}%")

    # 2. 创建优化器
    # <--- 只需为教师模型创建优化器
    optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=LEARNING_RATE)
    
    # 3. 打印实验配置信息
    print("\n--- 开始微调教师模型 (ViT-Base Fine-tuning) ---")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}")
    
    # 4. 执行训练和评估循环
    for epoch in range(EPOCHS):
        avg_loss = finetune_epoch(teacher_model, train_loader, optimizer)
        accuracy = evaluate(teacher_model, test_loader)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  平均训练损失 (Average Train Loss): {avg_loss:.4f}")
        print(f"  测试集准确率 (Test Accuracy): {accuracy:.2f}%")
        
    # 5. 保存训练好的教师模型权重
    try:
        torch.save(teacher_model.state_dict(), MODEL_SAVE_PATH)
        print(f"\n--- 教师模型微调完成! ---")
        print(f"✅ 训练好的教师模型已保存到: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"⚠️ 模型保存失败: {e}")