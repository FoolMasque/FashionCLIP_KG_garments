import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from create_kg import get_graph_data
from torch_geometric.data import Data

class FashionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.G, self.all_entities, self.node_features = get_graph_data()
        self.class_fi = {}

    def generate_prompt(self, task_type, **kwargs):
        prompts = {
            'title_classification': 'The label of {gender},{coarse_category},{fine_category},{materials},{features},{colour},{sleeve_length},{closure},{fabric_type},{neckline},{pattern}',
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

        # attributes
        # 单标签属性
        closure = self.data_frame.iloc[idx]['closure']
        colour = self.data_frame.iloc[idx]['colour']
        fabric_type = self.data_frame.iloc[idx]['fabric Type']
        neckline = self.data_frame.iloc[idx]['neckline']
        pattern = self.data_frame.iloc[idx]['pattern']
        sleeve_length = self.data_frame.iloc[idx]['sleeve Length']

        # Multi-label classification for features, materials, and accents
        features = self.data_frame.iloc[idx]['features'].split(', ')
        materials = self.data_frame.iloc[idx]['materials'].split(', ')
        title = self.data_frame.iloc[idx]['title']
        attributes = {
            'features': features,
            'materials': materials,
            'closure': closure,
            'colour': colour,
            'fabric_type': fabric_type,
            'neckline': neckline,
            'pattern': pattern,
            'sleeve_length': sleeve_length,
            'title':title
        }
        nodes_in_subgraph = []

        for item in attributes.keys():
            entity_list = attributes[item]
            if isinstance(entity_list, list):
                for entity in entity_list:
                    try:
                        neighbors = list(self.G.neighbors(entity))
                        neighbors = neighbors + [entity]
                        nodes_in_subgraph += neighbors
                    except:
                        pass
                        #print(entity,"is not in the graph")
            else:
                try:
                    neighbors = list(self.G.neighbors(entity_list))
                    neighbors = neighbors + [entity_list]
                    nodes_in_subgraph += neighbors
                except:
                    pass
                    #print(entity_list,"is not in the graph")

        subgraph = self.G.subgraph(nodes_in_subgraph)
        x = []
        for node in subgraph.nodes():
            x = [self.all_entities.index(node)]
            break
        #x = [self.all_entities.index(node) for node in subgraph.nodes()]\

        edge_index = []

        if x == []:
            edge_index = [[0, 1],[1, 0]]
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            data = Data(x= torch.tensor([0], dtype=torch.float), edge_index=edge_index)
        else:
            x = torch.tensor(x, dtype=torch.float)
            for u, v in subgraph.edges():
                edge_index = [[self.all_entities.index(u), self.all_entities.index(v)],[self.all_entities.index(v), self.all_entities.index(u)]]
                break
            if edge_index == []:
                edge_index = [[0, 1],[1, 0]]
                x = torch.tensor([0], dtype=torch.float)
            # 转换为 PyTorch 张量
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            data = Data(x=x, edge_index=edge_index)


        gender_prompt = self.generate_prompt('gender_classification', gender=gender)
        coarse_category_prompt = self.generate_prompt('coarse_category_classification', gender=gender,
                                                      coarse_category=category)
        fine_category_prompt = self.generate_prompt('fine_category_classification', gender=gender,
                                                    coarse_category=category, fine_category=item_name)
        title_prompt = self.generate_prompt('title_classification', gender=gender,coarse_category=category,fine_category=item_name,**attributes)
        attribute_prompts = self.generate_prompt('attribute_classification', **attributes)
        label_map = {}
        # Store all prompts in a dictionary
        all_category_text_prompts = {
            'gender_classification': gender_prompt['gender_classification'],
            # 'title_classification': title_prompt['title_classification'],
            'coarse_category_classification': coarse_category_prompt['coarse_category_classification'],
            'fine_category_classification': fine_category_prompt['fine_category_classification'],
        }
        all_category_text_prompts.update(attribute_prompts)
        for key, value in all_category_text_prompts.items():
            if key not in self.class_fi.keys():
                self.class_fi[key] = {}
            if value not in self.class_fi[key] and len(self.class_fi[key]) == 0:
                self.class_fi[key][value] = 0
            elif value not in self.class_fi[key] and len(self.class_fi[key]) > 0:
                self.class_fi[key][value] = max(self.class_fi[key].values()) + 1

        for key, value in all_category_text_prompts.items():
            label_map[key] = []
            label_map[key].append(self.class_fi[key][value])

        sample = {
            'image': image,
            'all_category_text_prompts': all_category_text_prompts,
            'label_map': label_map,
            'data': data
        }
        return sample