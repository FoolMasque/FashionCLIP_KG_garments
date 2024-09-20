import torch
from Project.fashion.final_GraphKG.ttkey import extract_key_info
# from Project.fashion.final_GraphKG.est_with_knn import get_nerb
from loss import compute_sdm
import torch.nn as nn
import difflib


def find_closest_entity(entity, all_entities, cutoff=0.8):
    # Finding the most similar entity to entity with difflib
    close_matches = difflib.get_close_matches(entity, all_entities, n=1, cutoff=cutoff)
    if close_matches:
        return close_matches[0]  # Returns the most similar match
    return None


import torch
import torch.nn.functional as F


def find_closest_entity_by_embedding(entity_embedding, all_entity_embeddings, top_k=1):
    # Calculate the cosine similarity between the input entity embedding and all entity embeddings
    similarities = F.cosine_similarity(entity_embedding.unsqueeze(0), all_entity_embeddings, dim=-1)
    top_k_indices = torch.topk(similarities, top_k).indices
    return top_k_indices  # Returns the index of the most similar node


def get_nerb(entity_list, G, entity_embeddings, top_k=5):
    nodes_in_subgraph = []
    if isinstance(entity_list, list):
        for entity in entity_list:
            try:
                # Find the corresponding entity embedding by entity name
                entity_embedding = entity_embeddings.get(entity, None)
                if entity_embedding is None:
                    # If the embedding is not found, the entity is skipped
                    continue

                # Get neighbour nodes and limit the number to top_k
                neighbors = list(G.neighbors(entity))[:top_k]
                neighbors = neighbors + [entity]
                nodes_in_subgraph += neighbors
            except:
                pass
    else:
        try:
            entity_embedding = entity_embeddings.get(entity_list, None)
            if entity_embedding is None:
                return nodes_in_subgraph  # Returns null if no embedding is found
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
        # Step 1: Get image embedding Q
        image_inputs = self.processor(images=images, return_tensors="pt").to(device)
        image_embeds = self.clip_model.get_image_features(**image_inputs)

        gnn_feature = self.gnn_model(data)
        gnn_feature = self.gcn_align(gnn_feature)

        # Step 2: Get text embedding T
        all_category_text_embeds = {}
        contrastive_loss_dict = {}
        final_text_embeds = {}
        for key, text_prompts in all_category_text_prompts.items():
            text_prompt_list = all_category_text_prompts[key]
            # print(text_prompt)
            text_inputs = self.processor(text=text_prompt_list, return_tensors="pt", padding=True, truncation=True).to(device)
            text_embeds = self.clip_model.get_text_features(**text_inputs)
            all_category_text_embeds[key] = text_embeds

            # Step 3: Use the get_nerb function to get neighbouring nodes for each text separately
            neighbor_embeds_list = []
            for text_prompt in text_prompt_list:
                # print(text_prompt)
                top_k_neighbors = get_nerb(text_prompt, G, entity_embeddings=gnn_feature, top_k=top_k)
                # print(top_k_neighbors)
                # TODO
                # Step 4: Get the corresponding GNN embedding based on neighbouring nodes
                if top_k_neighbors:
                    neighbor_embeds = gnn_feature[top_k_neighbors]  # Get the corresponding GNN embedding based on the neighbour node index
                else:
                    # Use the average embedding of all nodes
                    neighbor_embeds = gnn_feature.mean(dim=0, keepdim=True).to(device)
                neighbor_embeds_list.append(torch.mean(neighbor_embeds, dim=0, keepdim=True))  # Average aggregated neighbour embedding
            # print(neighbor_embeds_list)
            # Step 5: Stack all text neighbour embeddings and blend them with the corresponding text embedding
            if neighbor_embeds_list:
                aggregated_gnn_embeds = torch.cat(neighbor_embeds_list, dim=0)  # Splice neighbour embedding of all text
            else:
                aggregated_gnn_embeds = torch.zeros_like(text_embeds).to(device)

            # Step 6: Fusion of computational text embeddings with GNN embeddings
            final_embeds = (text_embeds + aggregated_gnn_embeds) / 2  # Simply averaging as a fusion method
            final_text_embeds[key] = final_embeds

            contrastive_loss = compute_sdm(image_embeds, final_text_embeds[key], label[key][0].cuda(), 1 / 0.02)
            # print(contrastive_loss)
            contrastive_loss_dict[key] = contrastive_loss

        # Step 7: Compute the learning loss of image embeddings compared to GNN embeddings
        contrastive_loss_gnn = compute_sdm(image_embeds, gnn_feature, torch.arange(0, len(image_embeds)).cuda(), 1 / 0.02)

        outputs_dict = {
            "total_contrastive_loss": contrastive_loss_dict,
            "all_category_text_embeds": all_category_text_embeds,
            "gnn_loss": contrastive_loss_gnn
        }
        # Clean up unnecessary tensors to free up video memory
        del image_embeds, text_embeds
        torch.cuda.empty_cache()
        return outputs_dict

