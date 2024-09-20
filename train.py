import random
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import json
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.graphgym import GNN
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from torch_geometric.data import Data
from gnn import GCN
from dataset import FashionDataset
from loss import compute_sdm
from model import FashionClassificationModel

device = "cuda" if torch.cuda.is_available() else "cpu"
from torch_geometric.data import DataLoader
from create_kg import get_graph_data
from torchmetrics.functional import accuracy

# 保存损失和准确率的列表
train_loss_list = []
train_acc_list = []
test_acc_list = []

def train(model, dataloader, optimizer, G, epoch, test_loader, top_k=5):
    model.train()
    total_acc = {}
    avg_loss = 0
    for batch in dataloader:
        all_prompts = batch['all_category_text_prompts']
        for i in all_prompts:
            if i not in total_acc.keys():
                total_acc[i] = 0
        images = batch['image'].to(device)
        optimizer.zero_grad()
        labels = batch['label_map']
        with autocast():
            outputs = model(all_category_text_prompts=all_prompts,
                            images=images,label=labels,data=batch['data'].cuda(),G=G, top_k=top_k)
            total_contrastive_loss = 0
            for key, loss in outputs["total_contrastive_loss"].items():
                total_contrastive_loss += loss

            total_loss = total_contrastive_loss + outputs["gnn_loss"]

        # 计算准确率
        model.eval()
        with torch.no_grad():
            for key in all_prompts:
                sim_f = outputs['all_category_text_embeds'][key]
                sim_f = F.normalize(sim_f, dim=-1)
                similarity_scores = sim_f @ sim_f.t()

                pred_labels = similarity_scores.argmax(dim=1)
                acc = (pred_labels == labels[key][0].cuda()).float().mean()
                total_acc[key] += acc.item()
        # 反向传播和优化
        total_loss.backward()
        avg_loss += total_loss.item()
        optimizer.step()

    # 记录训练损失
    avg_loss = avg_loss / len(dataloader)
    train_loss_list.append(avg_loss)

    # 打印训练损失
    print(f"Epoch {epoch}, Training Loss: {avg_loss:.4f}")

    # 记录每个类别的平均训练准确率
    for key, acc in total_acc.items():
        avg_acc = acc / len(dataloader)
        train_acc_list.append((key, avg_acc))
        print(f"Training item: {key}, total_acc: {avg_acc:.4f}")

    # 进行测试评估
    dest(model, test_loader, G, top_k)


def dest(model, test_loader, G, top_k):
    model.eval()
    total_acc_1 = {}
    feature_ = {}
    label = {}
    from metric import rank
    with torch.no_grad():
        for batch in test_loader:
            all_prompts = batch['all_category_text_prompts']
            for i in all_prompts:
                if i not in total_acc_1.keys():
                    total_acc_1[i] = 0
            images = batch['image'].to(device)
            labels = batch['label_map']
            outputs = model(all_category_text_prompts=all_prompts,
                            images=images,label=labels,data=batch['data'].cuda(),G=G, top_k=top_k)

            for key in all_prompts:
                for i in all_prompts[key]:
                    if key not in label.keys():
                        label[key] = []
                    label[key].append(test_loader.dataset.dataset.class_fi[key][i])

                if key not in feature_.keys():
                    feature_[key] = []
                sim_f = outputs['all_category_text_embeds'][key]
                feature_[key].append(sim_f)
                sim_f = F.normalize(sim_f, dim=-1)

                similarity_scores = sim_f @ sim_f.t()
                pred_labels = similarity_scores.argmax(dim=1)
                acc = (pred_labels == labels[key][0].cuda()).float().mean()
                total_acc_1[key] += acc.item()

    # 记录测试结果
    for key in feature_.keys():
        feature_[key] = torch.cat(feature_[key], dim=0)
        label[key] = torch.tensor(label[key])
        all_cmc, mAP, mINP, indices = rank(feature_[key] @ feature_[key].t(), label[key], label[key])
        print("key:", {key}, "all_cmc:", all_cmc.numpy() / 100, "mAP", mAP / 100)
        test_acc_list.append((key, all_cmc.numpy() / 100, mAP / 100))

    # 打印测试准确率
    for key, acc in total_acc_1.items():
        avg_acc = acc / len(test_loader)
        print(f"Testing item: {key}, total_acc: {avg_acc:.4f}")

def save_results():
    # 保存训练和测试结果到文件
    if not os.path.exists('results'):
        os.makedirs('results')

    # 保存训练损失
    with open('results/train_loss.txt', 'w') as f:
        for loss in train_loss_list:
            f.write(f"{loss}\n")

    # 保存训练准确率
    with open('results/train_acc.txt', 'w') as f:
        for key, acc in train_acc_list:
            f.write(f"{key}: {acc}\n")

    # 保存测试结果
    with open('results/test_results.txt', 'w') as f:
        for key, cmc, mAP in test_acc_list:
            f.write(f"{key}: CMC: {cmc}, mAP: {mAP}\n")


def plot_loss_curve():
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_loss_list)), train_loss_list, label='Training Loss', marker='o', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('results/loss_curve.png')
    plt.show()


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Transformations for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])
    # Create the dataset and dataloader
    csv_file = 'test_processed_main_data.csv'
    img_dir = 'E:/dev/Project/fashion/Data/images'

    fashion_dataset = FashionDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)

    # 数据集长度
    dataset_size = len(fashion_dataset)

    # 定义训练集和测试集的大小
    test_size = int(0.2 * dataset_size)
    train_size = dataset_size - test_size

    # 使用 random_split 划分数据集
    train_dataset, test_dataset = random_split(fashion_dataset, [train_size, test_size])

    # 创建 DataLoader 仅用于测试集
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 初始化模型和处理器
    clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    # 初始化模型
    gnn_model = GCN(47, 47, 47)

    # 冻结浅层参数，只微调高层
    for name, param in clip_model.named_parameters():
        if "encoder.layers.7" in name or "encoder.layers.8" in name or "encoder.layers.9" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model = FashionClassificationModel(clip_model, processor, gnn_model=gnn_model, use_fusion=False).to(device)
    # 获取知识图谱 G
    G, all_entities, node_features = get_graph_data()
    # 定义损失函数
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    # 训练模型
    num_epochs = 5

    # 设置初始 alpha 值
    initial_alpha = 0.5  # 初始时对比学习损失和分类损失的权重相等
    for epoch in range(num_epochs):
        # 动态调整 alpha 值，可以根据具体情况调整策略
        # if epoch > 5:
        #     alpha = 0.3  # 后期更注重分类损失
        # else:
        #     alpha = initial_alpha
        train(model, train_dataloader, optimizer, G, epoch,test_loader,top_k=5)
        torch.save(model.state_dict(), f"fashion_mode_select1.pth")
    # 保存训练和测试结果
    save_results()
    # 绘制损失曲线
    plot_loss_curve()
