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

# Keep a list of losses and accuracy rates
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

        # Calculation accuracy
        model.eval()
        with torch.no_grad():
            for key in all_prompts:
                sim_f = outputs['all_category_text_embeds'][key]
                sim_f = F.normalize(sim_f, dim=-1)
                similarity_scores = sim_f @ sim_f.t()

                pred_labels = similarity_scores.argmax(dim=1)
                acc = (pred_labels == labels[key][0].cuda()).float().mean()
                total_acc[key] += acc.item()
        # Backpropagation and optimisation
        total_loss.backward()
        avg_loss += total_loss.item()
        optimizer.step()

    # Recording training losses
    avg_loss = avg_loss / len(dataloader)
    train_loss_list.append(avg_loss)

    # Print training loss
    print(f"Epoch {epoch}, Training Loss: {avg_loss:.4f}")

    # 记录每个类别的平均训练准确率
    for key, acc in total_acc.items():
        avg_acc = acc / len(dataloader)
        train_acc_list.append((key, avg_acc))
        print(f"Training item: {key}, total_acc: {avg_acc:.4f}")

    # Conducting test evaluations
    test(model, test_loader, G, top_k)


def test(model, test_loader, G, top_k):
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

    # Recording test results
    for key in feature_.keys():
        feature_[key] = torch.cat(feature_[key], dim=0)
        label[key] = torch.tensor(label[key])
        all_cmc, mAP, mINP, indices = rank(feature_[key] @ feature_[key].t(), label[key], label[key])
        print("key:", {key}, "all_cmc:", all_cmc.numpy() / 100, "mAP", mAP / 100)
        test_acc_list.append((key, all_cmc.numpy() / 100, mAP / 100))

    # Print Test Accuracy
    for key, acc in total_acc_1.items():
        avg_acc = acc / len(test_loader)
        print(f"Testing item: {key}, total_acc: {avg_acc:.4f}")

def save_results():
    # Save training and test results to file
    if not os.path.exists('results'):
        os.makedirs('results')

    # Preservation of training losses
    with open('results/train_loss.txt', 'w') as f:
        for loss in train_loss_list:
            f.write(f"{loss}\n")

    # Preservation of training accuracy
    with open('results/train_acc.txt', 'w') as f:
        for key, acc in train_acc_list:
            f.write(f"{key}: {acc}\n")

    # Saving test results
    with open('results/test_results.txt', 'w') as f:
        for key, cmc, mAP in test_acc_list:
            f.write(f"{key}: CMC: {cmc}, mAP: {mAP}\n")


def plot_loss_curve():
    # Plotting training loss curves
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

    # Data set length
    dataset_size = len(fashion_dataset)

    # Define the size of the training and test sets
    test_size = int(0.2 * dataset_size)
    train_size = dataset_size - test_size

    # Splitting a dataset using random_split, also can use dataloader split
    train_dataset, test_dataset = random_split(fashion_dataset, [train_size, test_size])

    # Creating a DataLoader for Test Sets Only
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialising models and processors
    clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    # Initialising the model
    gnn_model = GCN(47, 47, 47)

    # Freeze shallow parameters and fine-tune only the upper levels
    for name, param in clip_model.named_parameters():
        if "encoder.layers.7" in name or "encoder.layers.8" in name or "encoder.layers.9" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model = FashionClassificationModel(clip_model, processor, gnn_model=gnn_model, use_fusion=False).to(device)
    # get Knowledge Graph G
    G, all_entities, node_features = get_graph_data()
    # Define the loss function
    # Optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    # train
    num_epochs = 5

    # Set initial alpha value
    initial_alpha = 0.5  # Initially the comparison learning loss and classification loss are equally weighted
    for epoch in range(num_epochs):
        # Dynamically adjusting the alpha value allows you to tailor the strategy to the situation
        # if epoch > 5:
        #     alpha = 0.3  # More focus on classified losses in later years
        # else:
        #     alpha = initial_alpha
        train(model, train_dataloader, optimizer, G, epoch,test_loader,top_k=5)
        torch.save(model.state_dict(), f"fashion_mode_select1.pth")
    # Save training and test results
    save_results()
    # Plotting loss curves
    plot_loss_curve()
