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

device = "cuda" if torch.cuda.is_available() else "cpu"


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

    def generate_prompt(self, task_type, **kwargs):
        prompts = {
            'gender_classification': "This item is {gender}'s clothing.",
            'coarse_category_classification': "This is a {gender} {coarse_category}.",
            'fine_category_classification': "This {gender} {coarse_category} is a {fine_category}.",
            'attribute_classification': {
                'materials': "This clothing item is made of: {materials}.",
                'features': "This clothing item has the following features: {features}.",
                'colour': "The color of this clothing item is {colour}.",
                'sleeve_length': "The sleeve length of this clothing item is {sleeve_length}.",
                'closure': "The closure type of this clothing item is {closure}.",
                'fabric_type': "The fabric type of this clothing item is {fabric_type}.",
                'neckline': "The neckline of this clothing item is {neckline}.",
                'pattern': "The pattern on this clothing item is {pattern}."
            }
        }

        if task_type in prompts:
            if task_type == 'attribute_classification':
                attribute_prompts = {}
                for attribute, prompt_template in prompts[task_type].items():
                    if attribute in kwargs:
                        if isinstance(kwargs[attribute], list):
                            kwargs[attribute] = ', '.join(kwargs[attribute])
                        attribute_prompts[f"{attribute}_classification"] = prompt_template.format(**kwargs)
                return attribute_prompts
            else:
                return {task_type: prompts[task_type].format(**kwargs)}
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    # 示例使用
    # gender_prompt = generate_prompt('gender_classification', gender='women')
    # print(gender_prompt)  # 输出: This item is women's clothing.
    #
    # coarse_category_prompt = generate_prompt('coarse_category_classification', gender='women', coarse_category='dress')
    # print(coarse_category_prompt)  # 输出: This is a women dress.
    #
    # fine_category_prompt = generate_prompt('fine_category_classification', gender='women', coarse_category='dress',
    #                                        fine_category='maxi dress')
    # print(fine_category_prompt)  # 输出: This women dress is a maxi dress.
    #
    # color_attribute_prompt = generate_prompt('attribute_classification', attribute='color', color='blue')
    # print(color_attribute_prompt)  # 输出: The color of this clothing item is blue.
    #
    # sleeve_length_attribute_prompt = generate_prompt('attribute_classification', attribute='sleeve_length',
    #                                                  sleeve_length='long')
    # print(sleeve_length_attribute_prompt)  # 输出: The sleeve length of this clothing item is long.

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = f"{self.img_dir}/{self.data_frame.iloc[idx]['sku'].strip()}_image1.jpg"
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load labels
        gender = self.data_frame.iloc[idx]['gender']
        category = self.data_frame.iloc[idx]['ebay Category']
        item_name = self.data_frame.iloc[idx]['item Type']

        gender_label = self.mappings['hierarchical']['gender_mapping'][gender]
        category_label = self.mappings['hierarchical']['category_mapping'][category]
        item_name_label = self.mappings['hierarchical']['item_name_mapping'][item_name]

        # attributes
        # 单标签属性
        closure = self.data_frame.iloc[idx]['closure']
        colour = self.data_frame.iloc[idx]['colour']
        fabric_type = self.data_frame.iloc[idx]['fabric Type']
        neckline = self.data_frame.iloc[idx]['neckline']
        pattern = self.data_frame.iloc[idx]['pattern']
        sleeve_length = self.data_frame.iloc[idx]['sleeve Length']

        closure_label = self.mappings['attributes']['closure'][closure]
        colour_label = self.mappings['attributes']['colour'][colour]
        fabric_type_label = self.mappings['attributes']['fabric_type'][fabric_type]
        neckline_label = self.mappings['attributes']['neckline'][neckline]
        pattern_label = self.mappings['attributes']['pattern'][pattern]
        sleeve_length_label = self.mappings['attributes']['sleeve_length'][sleeve_length]

        # Multi-label classification for features, materials, and accents
        features = self.data_frame.iloc[idx]['features'].split(', ')
        materials = self.data_frame.iloc[idx]['materials'].split(', ')

        feature_labels = [self.mappings['attributes']['features'][feat] for feat in features]
        material_labels = [self.mappings['attributes']['materials'][mat] for mat in materials]
        feature_vector = [0] * len(self.mappings['attributes']['features'])
        for index in feature_labels:
            feature_vector[index] = 1

        material_vector = [0] * len(self.mappings['attributes']['materials'])
        for index in material_labels:
            material_vector[index] = 1
        attributes = {
            'features': features,
            'materials': materials,
            'closure': closure,
            'colour': colour,
            'fabric_type': fabric_type,
            'neckline': neckline,
            'pattern': pattern,
            'sleeve_length': sleeve_length
        }

        # 生成文本 prompt
        # Generate text prompts for each classification task
        gender_prompt = self.generate_prompt('gender_classification', gender=gender)
        coarse_category_prompt = self.generate_prompt('coarse_category_classification', gender=gender,
                                                      coarse_category=category)
        fine_category_prompt = self.generate_prompt('fine_category_classification', gender=gender,
                                                    coarse_category=category, fine_category=item_name)

        attribute_prompts = self.generate_prompt('attribute_classification', **attributes)

        # Store all prompts in a dictionary
        all_category_text_prompts = {
            'gender_classification': gender_prompt['gender_classification'],
            'coarse_category_classification': coarse_category_prompt['coarse_category_classification'],
            'fine_category_classification': fine_category_prompt['fine_category_classification'],
        }
        all_category_text_prompts.update(attribute_prompts)

        sample = {
            'image': image,
            'all_category_text_prompts': all_category_text_prompts,
            'gender_label': torch.tensor(gender_label),
            'category_label': torch.tensor(category_label),
            'item_name_label': torch.tensor(item_name_label),

            'feature_labels': torch.tensor(feature_vector),
            'material_labels': torch.tensor(material_vector),

            'closure_label': torch.tensor(closure_label),
            'colour_label': torch.tensor(colour_label),
            'fabric_type_label': torch.tensor(fabric_type_label),
            'neckline_label': torch.tensor(neckline_label),
            'pattern_label': torch.tensor(pattern_label),
            'sleeve_length_label': torch.tensor(sleeve_length_label)
        }
        return sample


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, image_embeds, text_embeds):
        # Normalize embeddings
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Compute similarity
        logits = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        # logits = self.cosine_similarity(image_embeds.unsqueeze(1), text_embeds.unsqueeze(0)) / self.temperature
        labels = torch.arange(logits.size(0)).long().to(image_embeds.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class FashionClassificationModel(nn.Module):
    def __init__(self, clip_model, processor, mappings, use_fusion=False):
        super(FashionClassificationModel, self).__init__()
        self.clip_model = clip_model
        self.processor = processor
        self.mappings = mappings
        self.contrastive_loss_fn = ContrastiveLoss()
        self.use_fusion = use_fusion

        # 可训练的 prompt embeddings
        # 选择 category_mapping 作为 prompt embeddings 的基础是为了在对比学习和分类任务中增强对粗粒度类别的区分能力。
        # self.prompt_embeddings = nn.Parameter(
        #     torch.randn(len(mappings['hierarchical']['category_mapping']), clip_model.config.projection_dim))

        # 定义 new_prompt_embeddings
        self.gender_prompt_embeddings = nn.Parameter(torch.randn(1, clip_model.config.projection_dim))

        # 看clip_model.config，模型的文本和视觉部分的特征维度是分别由 text_config 和 vision_config 中的 hidden_size 决定的。
        # 要基于视觉特征进行分类，可以使用 vision_config.hidden_size；如果要基于文本特征，可以使用 text_config.hidden_size；如果要基于投影的对比学习特征，可以使用 projection_dim
        # 在初始化所有分类器时都应该使用 clip_model.config.projection_dim 作为特征维度
        # 定义层次分类器
        self.gender_classifier = nn.Linear(clip_model.config.projection_dim, len(mappings['hierarchical']['gender_mapping']))
        self.category_classifier = nn.Linear(clip_model.config.projection_dim, len(mappings['hierarchical']['category_mapping']))
        self.item_classifier = nn.Linear(clip_model.config.projection_dim, len(mappings['hierarchical']['item_name_mapping']))

        # 动态定义属性分类器
        self.attribute_classifiers = nn.ModuleDict()
        for attribute, classes in mappings['attributes'].items():
            # brand类别有点多，先可考虑不管
            if attribute == 'brand':
                continue
            num_classes = len(classes)
            self.attribute_classifiers[attribute] = nn.Linear(clip_model.config.projection_dim, num_classes)

    def initialize_prompt_embeddings(self):
        # 从预训练的文本嵌入初始化 new_prompt_embeddings
        with torch.no_grad():
            # 初始化性别分类的 prompt embeddings
            gender_prompts = ["This item is men's clothing.", "This item is women's clothing."]
            text_embeds = []
            for prompt in gender_prompts:
                text_inputs = self.processor(text=prompt, return_tensors="pt", padding=True, truncation=True).to(device)
                text_embeds.append(self.clip_model.get_text_features(**text_inputs))
            self.gender_prompt_embeddings.data = torch.mean(torch.stack(text_embeds), dim=0)

    def forward(self, all_category_text_prompts, images):

        # 处理图像输入，得到图像嵌入
        image_inputs = self.processor(images=images, return_tensors="pt").to(device)
        image_embeds = self.clip_model.get_image_features(**image_inputs)
        # 融合嵌入（如果使用）
        if self.use_fusion:
            gender_fused_embeds = image_embeds + self.gender_prompt_embeddings.expand_as(image_embeds)
            fused_embeds = image_embeds + gender_fused_embeds
        else:
            fused_embeds = image_embeds

        all_category_text_embeds = {}
        contrastive_loss_dict = {}
        for key, text_prompts in all_category_text_prompts.items():
            text_prompt = all_category_text_prompts[key]
            text_inputs = self.processor(text=text_prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            text_outputs = self.clip_model.get_text_features(**text_inputs)
            text_embeds = text_outputs
            all_category_text_embeds[key] = text_embeds

            # 计算对比学习损失
            contrastive_loss = self.contrastive_loss_fn(image_embeds, all_category_text_embeds[key])
            contrastive_loss_dict[key] = contrastive_loss

        # 性别分类输出
        # 看下只在gender加入trainable
        gender_fused_embeds = image_embeds + self.gender_prompt_embeddings.expand_as(image_embeds)
        fused_embeds0 = image_embeds + gender_fused_embeds

        gender_logits = self.gender_classifier(fused_embeds0)
        gender_probs = torch.softmax(gender_logits, dim=1)
        gender_predicted = torch.argmax(gender_probs, dim=1)
        gender_list = gender_predicted.tolist()

        # 将与当前任务无关的类别的 logits 设置为负无穷大来“冻结”这些类别。模型在进行 softmax 计算时忽略这些无关的类别，从而简化分类任务并提高分类准确性。
        # coarse_category_logits = torch.full_like(self.category_classifier(fused_embeds), float('-inf'))
        coarse_category_logits = torch.full_like(self.category_classifier(fused_embeds), 0.)

        # 根据 gender_predicted 动态设置 category_logits 的值
        for batch_idx, gender_index in enumerate(gender_list):
            relevant_coarse_indices = self.mappings['hierarchical']['nested_mapping'][str(gender_index)]
            for coarse_index in relevant_coarse_indices.keys():
                coarse_category_logits[batch_idx, int(coarse_index)] = self.category_classifier(fused_embeds)[
                    batch_idx, int(coarse_index)]

        # 将负无穷大值替换为一个非常小的负数
        # 在 softmax 计算中，如果 logits 包含负无穷大值，数学运算可能会变得不稳定，导致结果为无穷大（inf）或非数（NaN）。
        # 1e-9会导致运算中的溢出， float16 的表示范围大约是从 -65504 到 65504，超出会导致溢出
        # coarse_category_logits[coarse_category_logits == float('-inf')] = 0

        coarse_category_probs = torch.softmax(coarse_category_logits, dim=1)
        coarse_category_predicted = torch.argmax(coarse_category_probs, dim=1)
        coarse_category_list = coarse_category_predicted.tolist()

        fine_category_logits = torch.full_like(self.item_classifier(fused_embeds), 0.)
        # 根据 coarse_category_list 动态设置 category_logits 的值
        for batch_idx, (gender_index, coarse_index) in enumerate(zip(gender_list, coarse_category_list)):
            relevant_fine_indices = self.mappings['hierarchical']['nested_mapping'][str(gender_index)][str(coarse_index)]
            for fine_index in relevant_fine_indices:
                fine_category_logits[batch_idx, fine_index] = 0

        fine_category_probs = torch.softmax(fine_category_logits, dim=1)
        fine_category_predicted = torch.argmax(fine_category_probs, dim=1)
        fine_category_list = fine_category_predicted.tolist()

        # 属性分类输出
        attribute_logits = {}
        for attribute, classifier in self.attribute_classifiers.items():
            attribute_logits[attribute] = classifier(fused_embeds)

        outputs_dict = {
            "gender_logits": gender_logits,
            "coarse_category_logits": coarse_category_logits,
            "fine_category_logits": fine_category_logits,
            "total_contrastive_loss": contrastive_loss_dict,
            **attribute_logits
        }

        # 清理不必要的张量以释放显存
        del image_embeds, text_embeds, fused_embeds
        torch.cuda.empty_cache()

        return outputs_dict


def train(model, dataloader, criterion, optimizer, epoch, alpha=0.5):
    model.train()
    total_loss = 0

    for batch in dataloader:
        all_prompts = batch['all_category_text_prompts']
        images = batch['image'].to(device)
        gender_labels = batch['gender_label'].to(device)
        coarse_category_labels = batch['category_label'].to(device)
        fine_category_labels = batch['item_name_label'].to(device)
        closure_label = batch['closure_label'].to(device)
        colour_label = batch['colour_label'].to(device)
        fabric_type_label = batch['fabric_type_label'].to(device)
        neckline_label = batch['neckline_label'].to(device)
        pattern_label = batch['pattern_label'].to(device)
        sleeve_length_label = batch['sleeve_length_label'].to(device)
        feature_labels = batch['feature_labels'].to(device)
        material_labels = batch['material_labels'].to(device)

        # 前向传播

        optimizer.zero_grad()
        with autocast():
            outputs = model(all_category_text_prompts=all_prompts, images=images)
            total_contrastive_loss = 0
            total_classifier_loss = 0

            gender_logits = outputs["gender_logits"]
            loss_gender = criterion['gender'](gender_logits, gender_labels)
            total_contrastive_loss += outputs["total_contrastive_loss"]['gender_classification']
            total_classifier_loss += loss_gender
            # print(f"gender loss:{loss_gender}")

            # 计算粗粒度类别分类损失
            coarse_category_logits = outputs["coarse_category_logits"]
            loss_coarse_category = criterion['category'](coarse_category_logits, coarse_category_labels)
            total_contrastive_loss += outputs["total_contrastive_loss"]['coarse_category_classification']
            total_classifier_loss += loss_coarse_category

            # 计算细粒度类别分类损失
            fine_category_logits = outputs["fine_category_logits"]
            loss_fine_category = criterion['item'](fine_category_logits, fine_category_labels)
            total_contrastive_loss += outputs["total_contrastive_loss"]['fine_category_classification']
            total_classifier_loss += loss_fine_category

            # 计算属性分类损失
            mask = feature_labels != -1
            loss_features = criterion['features'](outputs['features'][mask], feature_labels[mask].float())
            total_contrastive_loss += outputs["total_contrastive_loss"]['features_classification']
            total_classifier_loss += loss_features

            if not (material_labels == -1).all():
                mask = material_labels != -1
                loss_materials = criterion['materials'](outputs['materials'][mask], material_labels[mask].float())
                total_contrastive_loss += outputs["total_contrastive_loss"]['materials_classification']
                total_classifier_loss += loss_materials

            if not (closure_label == -1).all():
                mask = closure_label != -1
                loss_closure = criterion['closure'](outputs['closure'][mask], closure_label[mask])
                total_contrastive_loss += outputs["total_contrastive_loss"]['closure_classification']
                total_classifier_loss += loss_closure

            if not (colour_label == -1).all():
                mask = colour_label != -1
                loss_colour = criterion['colour'](outputs['colour'][mask], colour_label[mask])
                total_contrastive_loss += outputs["total_contrastive_loss"]['colour_classification']
                total_classifier_loss += loss_colour

            if not (fabric_type_label == -1).all():
                mask = fabric_type_label != -1
                loss_fabric_type = criterion['fabric_type'](outputs['fabric_type'][mask], fabric_type_label[mask])
                total_contrastive_loss += outputs["total_contrastive_loss"]['fabric_type_classification']
                total_classifier_loss += loss_fabric_type

            if not (neckline_label == -1).all():
                mask = neckline_label != -1
                loss_neckline = criterion['neckline'](outputs['neckline'][mask], neckline_label[mask])
                total_contrastive_loss += outputs["total_contrastive_loss"]['neckline_classification']
                total_classifier_loss += loss_neckline

            if not (pattern_label == -1).all():
                mask = pattern_label != -1
                loss_pattern = criterion['pattern'](outputs['pattern'][mask], pattern_label[mask])
                total_contrastive_loss += outputs["total_contrastive_loss"]['pattern_classification']
                total_classifier_loss += loss_pattern

            if not (sleeve_length_label == -1).all():
                mask = sleeve_length_label != -1
                loss_sleeve_length = criterion['sleeve_length'](outputs['sleeve_length'][mask], sleeve_length_label[mask])
                total_contrastive_loss += outputs["total_contrastive_loss"]['sleeve_length_classification']
                total_classifier_loss += loss_sleeve_length

            # 总损失
            # total_loss = total_contrastive_loss + total_classifier_loss
            # 动态调整损失权重
            total_loss = alpha * total_contrastive_loss + (1 - alpha) * total_classifier_loss

        # 反向传播和优化
        total_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {total_loss.item()}")
    avg_loss = total_loss / len(dataloader)
    return avg_loss


# 定义预测函数
def predict(model, image, processor):
    model.eval()
    with torch.no_grad():
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        # TODO:模型输入的text问题
        outputs = model(images=image_inputs['pixel_values'])
        predictions = {
            "gender": torch.argmax(outputs["gender_logits"], dim=1).item(),
            "coarse_category": torch.argmax(outputs["coarse_category_logits"], dim=1).item(),
            "fine_category": torch.argmax(outputs["fine_category_logits"], dim=1).item(),
        }
        print(predictions)
    return predictions


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from {path}")
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化模型和处理器
    clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    # 读取映射文件
    mappings_file = 'E:/dev/Project/fashion/all_mappings.json'
    with open(mappings_file, 'r') as f:
        mappings = json.load(f)

    # 初始化模型
    model = FashionClassificationModel(clip_model, processor, mappings).to(device)

    # 加载保存的模型权重
    model = load_model(model, "fashion_model_epoch_9.pth")

    # 加载和预处理输入图像
    img_path = "dati.jpeg"
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image)

    # 进行预测
    predictions = predict(model, image, processor)
    print("Predictions:", predictions)


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

    # Create the dataset and dataloader
    csv_file = 'E:/dev/Project/fashion/test_processed_main_data.csv'
    img_dir = 'E:/dev/Project/fashion/Data/images'
    mappings_file = 'E:/dev/Project/fashion/all_mappings.json'

    fashion_dataset = FashionDataset(csv_file=csv_file, img_dir=img_dir, mappings_file=mappings_file,
                                     transform=transform)
    dataloader0 = DataLoader(fashion_dataset, batch_size=32, shuffle=True)

    # 初始化模型和处理器
    clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    # 读取映射文件
    with open(mappings_file, 'r') as f:
        mappings = json.load(f)
    # 初始化模型
    model = FashionClassificationModel(clip_model, processor, mappings, use_fusion=False).to(device)

    # 初始化 prompt embeddings
    model.initialize_prompt_embeddings()
    # 定义损失函数
    criterion = {
        "gender": nn.CrossEntropyLoss(),
        "category": nn.CrossEntropyLoss(),
        "item": nn.CrossEntropyLoss(),
        "features": nn.BCEWithLogitsLoss(),
        "materials": nn.BCEWithLogitsLoss(),
        "closure": nn.CrossEntropyLoss(),
        "colour": nn.CrossEntropyLoss(),
        "fabric_type": nn.CrossEntropyLoss(),
        "neckline": nn.CrossEntropyLoss(),
        "pattern": nn.CrossEntropyLoss(),
        "sleeve_length": nn.CrossEntropyLoss()
    }

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练模型
    num_epochs = 2
    # 设置初始 alpha 值
    initial_alpha = 0.5  # 初始时对比学习损失和分类损失的权重相等
    for epoch in range(num_epochs):
        # 动态调整 alpha 值，可以根据具体情况调整策略
        if epoch > 5:
            alpha = 0.3  # 后期更注重分类损失
        else:
            alpha = initial_alpha
        train(model, dataloader0, criterion, optimizer, epoch, alpha=alpha)

    torch.save(model.state_dict(), f"fashion_model.pth")

    # 进行预测

    # # 加载和预处理输入图像
    # img_path = "dati.jpeg"
    # image = Image.open(img_path).convert("RGB")
    # image = transform(image)
    # predictions = predict(model, image, processor)
    # print("Predictions:", predictions)

    # main()
