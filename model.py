
import torch

from Project.fashion.final_GraphKG.ttkey import extract_key_info
# from Project.fashion.final_GraphKG.est_with_knn import get_nerb
from loss import compute_sdm
import torch.nn as nn
import difflib


def find_closest_entity(entity, all_entities, cutoff=0.8):
    # 通过 difflib 找到与 entity 最相似的实体
    close_matches = difflib.get_close_matches(entity, all_entities, n=1, cutoff=cutoff)
    if close_matches:
        return close_matches[0]  # 返回最相似的匹配
    return None


import torch
import torch.nn.functional as F


def find_closest_entity_by_embedding(entity_embedding, all_entity_embeddings, top_k=1):
    # 计算输入实体嵌入与所有实体嵌入之间的余弦相似度
    similarities = F.cosine_similarity(entity_embedding.unsqueeze(0), all_entity_embeddings, dim=-1)
    top_k_indices = torch.topk(similarities, top_k).indices
    return top_k_indices  # 返回最相似的节点索引


def get_nerb(entity_list, G, entity_embeddings, top_k=5):
    nodes_in_subgraph = []
    if isinstance(entity_list, list):
        for entity in entity_list:
            try:
                # 通过实体名称找到对应的实体嵌入
                entity_embedding = entity_embeddings.get(entity, None)
                if entity_embedding is None:
                    # 如果找不到嵌入，则跳过该实体
                    continue

                # 获取邻居节点，并限制数量为 top_k
                neighbors = list(G.neighbors(entity))[:top_k]
                neighbors = neighbors + [entity]
                nodes_in_subgraph += neighbors
            except:
                pass
    else:
        try:
            entity_embedding = entity_embeddings.get(entity_list, None)
            if entity_embedding is None:
                return nodes_in_subgraph  # 如果找不到嵌入，则返回空
            neighbors = list(G.neighbors(entity_list))[:top_k]
            neighbors = neighbors + [entity_list]
            nodes_in_subgraph += neighbors
        except:
            pass
    return nodes_in_subgraph


class FashionClassificationModel(nn.Module):
    def __init__(self, clip_model, processor, gnn_model, use_fusion=False):
        super(FashionClassificationModel, self).__init__()
        self.clip_model = clip_model
        self.processor = processor
        self.gnn_model = gnn_model
        self.use_fusion = use_fusion
        self.gcn_align = nn.Linear(47, clip_model.config.projection_dim)

    def test(self, images, text, device='cuda'):
        with torch.no_grad():
            print(images)
            image_inputs = self.processor(images=images, return_tensors="pt").to(device)
            image_embeds = self.clip_model.get_image_features(**image_inputs).detach()
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
            text_outputs = self.clip_model.get_text_features(**text_inputs)
            outputs_dict = {
                "image_embeds": image_embeds,
                "text_outputs": text_outputs,
            }
            return outputs_dict

    def forward(self, all_category_text_prompts, images, label, data, G, top_k=5, device='cuda'):
        # Step 1: 获取图像嵌入 Q
        image_inputs = self.processor(images=images, return_tensors="pt").to(device)
        image_embeds = self.clip_model.get_image_features(**image_inputs)

        gnn_feature = self.gnn_model(data)
        gnn_feature = self.gcn_align(gnn_feature)

        # Step 2: 获取文本嵌入 T
        all_category_text_embeds = {}
        contrastive_loss_dict = {}
        final_text_embeds = {}
        for key, text_prompts in all_category_text_prompts.items():
            text_prompt_list = all_category_text_prompts[key]
            # print(text_prompt)
            text_inputs = self.processor(text=text_prompt_list, return_tensors="pt", padding=True, truncation=True).to(device)
            text_embeds = self.clip_model.get_text_features(**text_inputs)
            all_category_text_embeds[key] = text_embeds

            # Step 3: 使用 get_nerb 函数分别为每个文本获取邻居节点
            neighbor_embeds_list = []
            for text_prompt in text_prompt_list:
                # print(text_prompt)
                top_k_neighbors = get_nerb(text_prompt, G, entity_embeddings=gnn_feature, top_k=top_k)
                # print(top_k_neighbors)
                # Step 4: 根据邻居节点获取对应的 GNN 嵌入
                if top_k_neighbors:
                    neighbor_embeds = gnn_feature[top_k_neighbors]  # 根据邻居节点索引获取相应的 GNN 嵌入
                else:
                    # TODO
                    # 使用所有节点的平均嵌入
                    neighbor_embeds = gnn_feature.mean(dim=0, keepdim=True).to(device)
                neighbor_embeds_list.append(torch.mean(neighbor_embeds, dim=0, keepdim=True))  # 平均聚合邻居嵌入
            # print(neighbor_embeds_list)
            # Step 5: 将所有文本的邻居嵌入进行堆叠，并与对应的文本嵌入融合
            if neighbor_embeds_list:
                aggregated_gnn_embeds = torch.cat(neighbor_embeds_list, dim=0)  # 将所有文本的邻居嵌入拼接起来
            else:
                aggregated_gnn_embeds = torch.zeros_like(text_embeds).to(device)

            # Step 6: 计算文本嵌入与 GNN 嵌入的融合
            final_embeds = (text_embeds + aggregated_gnn_embeds) / 2  # 简单地取平均作为融合方式
            final_text_embeds[key] = final_embeds

            contrastive_loss = compute_sdm(image_embeds, final_text_embeds[key], label[key][0].cuda(), 1 / 0.02)
            # print(contrastive_loss)
            contrastive_loss_dict[key] = contrastive_loss

        # Step 6: 计算图像嵌入与 GNN 嵌入的对比学习损失
        contrastive_loss_gnn = compute_sdm(image_embeds, gnn_feature, torch.arange(0, len(image_embeds)).cuda(), 1 / 0.02)

        outputs_dict = {
            "total_contrastive_loss": contrastive_loss_dict,
            "all_category_text_embeds": all_category_text_embeds,
            "gnn_loss": contrastive_loss_gnn
        }
        # 清理不必要的张量以释放显存
        del image_embeds, text_embeds
        torch.cuda.empty_cache()
        return outputs_dict

