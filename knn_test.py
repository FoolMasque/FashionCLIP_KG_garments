import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
import json
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.graphgym import GNN
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from torch_geometric.data import Data

from dataset import FashionDataset
from loss import compute_sdm
from model import FashionClassificationModel

device = "cuda" if torch.cuda.is_available() else "cpu"
from torch_geometric.data import DataLoader
from create_kg import get_graph_data


def get_nerb(entity_list,G, top_k=5):
    nodes_in_subgraph = []
    if isinstance(entity_list, list):
        for entity in entity_list:
            try:
                # Limit the number of neighbouring nodes returned to top_k by slicing [:top_k]
                neighbors = list(G.neighbors(entity))[:top_k]
                neighbors = neighbors + [entity]
                nodes_in_subgraph += neighbors
            except:
                pass
    else:
        try:
            # Performing the same operation on a single entity restricts the neighbouring nodes returned to top_k
            neighbors = list(G.neighbors(entity_list))[:top_k]
            neighbors = neighbors + [entity_list]
            nodes_in_subgraph += neighbors
        except:
            pass
    return nodes_in_subgraph

from torchmetrics.functional import accuracy

def est0(model, image,top_k=5):
    model.train()
    G, all_entities, node_features = get_graph_data()

    # Ensure that node_features is in Tensor format
    node_feature_list = []
    for node in all_entities:
        feature = node_features[node]
        if isinstance(feature, list):  # If the feature is a list, convert it to a Tensor.
            feature = torch.tensor(feature, dtype=torch.float32)
        node_feature_list.append(feature)

    node_feature_tensor = torch.stack(node_feature_list).to(device)  # Splicing all node features into a tensor

    all_prompts = ['blue', 'black', 'red']

    # Encoding text as embedding vectors using the CLIP model
    text_inputs = model.processor(text=all_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    text_embeddings = model.clip_model.get_text_features(**text_inputs).detach()  # Getting Text Embedding with the CLIP Model

    with autocast():
        outputs = model.test(images=image,text=all_prompts)
    model.eval()
    with torch.no_grad():
        sim_i = outputs['image_embeds']
        sim_t = outputs['text_outputs']
        sim_i = F.normalize(sim_i, dim=-1)
        sim_t = F.normalize(sim_t, dim=-1)
        similarity_scores = sim_i @ sim_t.t()
        print("with out G",similarity_scores)

    g_n = {}
    for i in range(len(all_prompts)):
        item = all_prompts[i]
        g_n[i] = get_nerb(item, G, top_k=top_k)
        # Print the TopK neighbour nodes for each text description
        print(f"TopK neighbors for '{item}': {g_n[i]}")
    with torch.no_grad():
        sim_i = outputs['image_embeds']
        sim_t = outputs['text_outputs']
        sim_i = F.normalize(sim_i, dim=-1)
        sim_t = F.normalize(sim_t, dim=-1)
        similarity_scores = sim_i @ sim_t.t()
        print("with G", similarity_scores)
        print("second label", g_n)



def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from {path}")
    return model


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
        #transforms.Normalize((0.5,), (0.5,))
    ])
    # Create the dataset and dataloader
    csv_file = 'test_processed_main_data.csv'
    img_dir = 'E:/dev/Project/fashion/Data/images'

    fashion_dataset = FashionDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    dataloader0 = DataLoader(fashion_dataset, batch_size=64, shuffle=True)
    # Initialising models and processors
    clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    from gnn import GCN

    gnn_model = GCN(47, 47, 47)
    model = FashionClassificationModel(clip_model, processor,gnn_model=gnn_model, use_fusion=False).to(device)
    model.load_state_dict(torch.load('fashion_mode_select1.pth'))
    img = r"E:\dev\Project\fashion\test1_back.jpeg"
    img = Image.open(img).convert("RGB")
    img = transform_test(img)
    est0(model, img)

