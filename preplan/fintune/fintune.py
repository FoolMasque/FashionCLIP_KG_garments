import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
import json
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = "cuda" if torch.cuda.is_available() else "cpu"

# 对比学习loss有问题
def info_nce_loss(image_embeds, text_embeds, temperature=0.07):
    # 计算图像嵌入和文本嵌入之间的余弦相似度矩阵
    logits_per_image = torch.matmul(image_embeds, text_embeds.T) / temperature

    # 创建正确配对的标签
    labels = torch.arange(logits_per_image.size(0)).to(image_embeds.device)

    # 使用交叉熵损失
    loss = F.cross_entropy(logits_per_image, labels)

    return loss
class CustomContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(CustomContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_embeds, text_embeds):

        # Normalize embeddings
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Compute similarity scores
        logits_per_image = torch.matmul(image_embeds, text_embeds.T) / self.temperature

        # Labels for contrastive loss
        labels = torch.arange(image_embeds.size(0)).to(image_embeds.device)

        # Create mask to avoid positive samples being treated as negatives
        mask = torch.eye(image_embeds.size(0), device=image_embeds.device, dtype=torch.bool)
        #出现梯度消失梯度爆炸了
        logits_per_image = logits_per_image.masked_fill(mask, float(-1))

        # Calculate contrastive loss using cross-entropy
        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_image.T, labels)

        return (loss_image + loss_text) / 2


contrastive_loss = CustomContrastiveLoss()


class FashionDataset(Dataset):
    def __init__(self, csv_file, img_dir, mappings_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.mappings = self.load_mappings(mappings_file)

    def load_mappings(self, mappings_file):
        import json
        with open(mappings_file, 'r') as f:
            mappings = json.load(f)
        return mappings

    def generate_comprehensive_prompt(self, gender, category, item_name, colour, fabric_type, materials, features, sleeve_length, closure, neckline, pattern):
        prompt = (
            f"This clothing item is designed for {gender}. It is a {sleeve_length} {colour} {category} made from {fabric_type} "
            f"and {materials}. This item features a {neckline} neckline and a {closure} closure. "
            f"The pattern on this clothing is {pattern}. Additionally, it is described as a {item_name} with the following features: {features}."
        )
        return prompt

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = f"{self.img_dir}/{self.data_frame.iloc[idx]['sku'].strip()}_image1.jpg"
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load labels
        # gender = self.data_frame.iloc[idx]['gender']
        # category = self.data_frame.iloc[idx]['ebay Category']
        # item_name = self.data_frame.iloc[idx]['item Type']
        # # attributes
        # closure = self.data_frame.iloc[idx]['closure']
        # colour = self.data_frame.iloc[idx]['colour']
        # fabric_type = self.data_frame.iloc[idx]['fabric Type']
        # neckline = self.data_frame.iloc[idx]['neckline']
        # pattern = self.data_frame.iloc[idx]['pattern']
        # sleeve_length = self.data_frame.iloc[idx]['sleeve Length']
        # # Multi-label classification for features, materials, and accents
        # features = self.data_frame.iloc[idx]['features']
        # materials = self.data_frame.iloc[idx]['materials']
        # 先去掉.split(', ')

        title = self.data_frame.iloc[idx]['title']

        # all_text_prompt = self.generate_comprehensive_prompt(gender, category, item_name, colour, fabric_type, materials, features, sleeve_length, closure, neckline, pattern)
        sample = {
            'image': image,
            'text_prompt': title}
        return sample


class FashionClassificationModel(nn.Module):
    def __init__(self, clip_model, processor):
        super(FashionClassificationModel, self).__init__()
        self.clip_model = clip_model
        self.processor = processor

    def forward(self, all_text_prompt, images):
        # 处理图像输入，得到图像嵌入
        image_inputs = self.processor(images=images, return_tensors="pt").to(device)
        image_embeds = self.clip_model.get_image_features(**image_inputs)

        text_inputs = self.processor(text=all_text_prompt, return_tensors="pt", padding=True, truncation=True, max_length=70).to(device)
        text_embeds = self.clip_model.get_text_features(**text_inputs)

        # 计算图像嵌入与各个类别文本嵌入之间的相似度
        cosine_sim = torch.matmul(image_embeds, text_embeds.T)
        result = torch.argmax(cosine_sim, dim=1)

        return result, image_embeds, text_embeds


def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()
        images = batch['image']
        text_prompts = batch['text_prompt']

        result, image_embeds, text_embeds = model(text_prompts, images)
        # 计算 InfoNCE 损失
        loss = contrastive_loss(image_embeds, text_embeds)
        loss.backward()

        # 打印每层的梯度
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} - grad mean: {param.grad.mean().item()}, grad std: {param.grad.std().item()}")

        optimizer.step()  # 优化器更新

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    return avg_loss


def validate(model, dataloader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image']
            text_prompts = batch['text_prompt']

            result, image_embeds, text_embeds = model(text_prompts, images)
            loss = contrastive_loss(image_embeds, text_embeds)
            total_loss += loss.item()

            # Calculate accuracy
            predicted_labels = result
            correct_predictions += (predicted_labels == torch.arange(len(result)).to(device)).sum().item()
            total_samples += len(predicted_labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def load_model(model_path, device):
    # 初始化模型和处理器
    clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    # 初始化自定义分类模型
    model = FashionClassificationModel(clip_model, processor).to(device)

    # 加载保存的模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切换到评估模式
    return model


# 定义预测函数
def predict(model, image, text_prompts):
    model.eval()
    with torch.no_grad():
        predicted_labels, _, _ = model(text_prompts, image)
        predicted_texts = [text_prompts[pred.item()] for pred in predicted_labels]
    return predicted_texts,predicted_labels

def predict_demo():
    images_path = ["APO180324006_image1.jpg", "APO180324001_image1.jpg", "FO180324003_image1.jpg"]
    images = [Image.open(i).convert("RGB") for i in images_path]
    # category_list = ["Long Sleeve", "Sleeveless", "3/4 Sleeve", "Short Sleeve"]
    # text_prompts = [f"The sleeve_length of this clothing is {i}" for i in category_list]
    # category_list = ['Polyester', 'Cotton', 'Nylon', 'Wool', 'Virgin Wool', 'Viscose', 'Merino Wool', 'Polyamide', 'Rayon', 'Acrylic', 'Angora', 'Suede', 'Leather', 'Ramie', 'Triacetate',
    #                  'New Wool', 'Polyacrylic', 'Shearling', 'Silk', 'Polyurethane', 'Lambswool', 'Linen']
    # text_prompts = [f"The clothing item is made of {i}" for i in category_list]

    category_list = ["Silver", "Gold", "Turquoise", "Beige", "Black", "Off White", "Green", "Navy Blue", "Burgundy", "Lilac", "Maroon", "White", "Brown", "Yellow", "Pink", "Orange", "Grey", "Multicoloured", "Red", "Khaki", "Purple", "Blue", "unknown"]
    text_prompts = [f"A photo of {i} clothing item." for i in category_list]

    # 加载保存的模型
    model_path = 'best_model.pth'
    model = load_model(model_path, device)
    predicted_texts,predicted_labels = predict(model, images, text_prompts)
    cls = [category_list[index] for index in predicted_labels]
    # print(predicted_texts)
    print(cls)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Transformations for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])

    # Create the dataset and dataloader
    csv_file = 'E:/dev/Project/fashion/test_processed_main_data.csv'
    img_dir = 'E:/dev/Project/fashion/Data/images'
    mappings_file = 'E:/dev/Project/fashion/all_mappings.json'

    dataset = FashionDataset(csv_file=csv_file, img_dir=img_dir, mappings_file=mappings_file, transform=transform)

    # Split the dataset into train, validation, and test sets
    # train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    # test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Initialize the model and processor
    clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    # 冻结低层次的参数，只微调高层次的 Transformer blocks
    for name, param in clip_model.named_parameters():
        # TODO: 选择解冻层
        # if "encoder.layers.4" in name or "encoder.layers.5" in name or "encoder.layers.6" in name or "encoder.layers.7" in name:
        if "encoder.layers.8" in name or \
            "encoder.layers.9" in name or \
            "encoder.layers.6" in name or \
            "encoder.layers.7" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 初始化模型
    model = FashionClassificationModel(clip_model, processor).to(device)

    # 优化器
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # 只对需要微调的参数进行优化
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    # 设置学习率调度器，当验证损失在'patience'个epoch后不再下降时，学习率乘以0.1
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    # 训练模型
    num_epochs = 50
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer)
        val_loss, val_accuracy = validate(model, val_loader)
        # 调度器根据验证损失调整学习率
        scheduler.step(val_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')


if __name__ == '__main__':
    # train
    main()

    # 测试
    # predict_demo()
