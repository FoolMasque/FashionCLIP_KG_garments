<<<<<<< HEAD
import rdflib
import networkx as nx
import matplotlib.pyplot as plt
import re


def get_graph_data():
    # 载入颜色知识图谱
    g = rdflib.Graph()
    g.parse("ontology.ttl", format="ttl")

    # 创建 NetworkX 图来表示知识图谱
    G = nx.Graph()

    # 解析类别文件
    category_file = "list_category_cloth.txt"
    with open(category_file, "r") as f:
        lines = f.readlines()
        category_count = int(lines[0].strip())  # 获取类别数量
        categories = [line.split() for line in lines[2:]]  # 从第三行开始读取

    # 解析属性文件
    attribute_file = "list_attr_cloth1.txt"
    with open(attribute_file, "r") as f:
        lines = f.readlines()
        attribute_count = int(lines[0].strip())  # 获取属性数量
        attributes = [re.split(r'\s{2,}', line.strip()) for line in lines[2:]]  # 从第三行开始读取

    # 将服装类别加入知识图谱
    for category_name, category_type in categories:
        G.add_node(category_name, type="Clothing Category", category_type=category_type)

    # 将服装属性加入知识图谱
    for attribute_name, attribute_type in attributes:
        G.add_node(attribute_name, type="Clothing Attribute", attribute_type=attribute_type)

    # 定义更加全面的连接规则
    for category_name, category_type in categories:
        for attribute_name, attribute_type in attributes:
            # 上半身衣物规则
            if category_type == "1":  # 上半身衣物
                if attribute_type == "1":  # 材质
                    G.add_edge(category_name, attribute_name, relation="has_texture")
                elif attribute_type == "2":  # 面料
                    G.add_edge(category_name, attribute_name, relation="has_fabric")
                elif attribute_type == "3":  # 形状
                    G.add_edge(category_name, attribute_name, relation="has_shape")
                elif attribute_type == "4":  # 部位
                    G.add_edge(category_name, attribute_name, relation="has_part")
                elif attribute_type == "5":  # 风格
                    G.add_edge(category_name, attribute_name, relation="has_style")

            # 下半身衣物规则
            elif category_type == "2":  # 下半身衣物
                if attribute_type == "1":  # 材质
                    G.add_edge(category_name, attribute_name, relation="has_texture")
                elif attribute_type == "2":  # 面料
                    G.add_edge(category_name, attribute_name, relation="has_fabric")
                elif attribute_type == "3":  # 形状
                    G.add_edge(category_name, attribute_name, relation="has_shape")
                elif attribute_type == "4":  # 部位
                    G.add_edge(category_name, attribute_name, relation="has_part")
                elif attribute_type == "5":  # 风格
                    G.add_edge(category_name, attribute_name, relation="has_style")

            # 全身衣物规则
            elif category_type == "3":  # 全身衣物
                if attribute_type == "1":  # 材质
                    G.add_edge(category_name, attribute_name, relation="has_texture")
                elif attribute_type == "2":  # 面料
                    G.add_edge(category_name, attribute_name, relation="has_fabric")
                elif attribute_type == "3":  # 形状
                    G.add_edge(category_name, attribute_name, relation="has_shape")
                elif attribute_type == "4":  # 部位
                    G.add_edge(category_name, attribute_name, relation="has_part")
                elif attribute_type == "5":  # 风格
                    G.add_edge(category_name, attribute_name, relation="has_style")

    # 整合颜色知识图谱（从颜色本体中获取颜色节点，并与服装类别、属性连接）
    namespace_manager = g.namespace_manager

    def remove_prefix(uri):
        uri_str = str(uri)
        for prefix, namespace in namespace_manager.namespaces():
            if uri_str.startswith(str(namespace)):
                return uri_str[len(str(namespace)):]
        return uri_str

    for subj, pred, obj in g:
        subj_filtered = remove_prefix(subj)
        pred_filtered = remove_prefix(pred)
        obj_filtered = remove_prefix(obj)

        # 处理 hasColourName 和 isOfGroup 关系
        if pred_filtered == "hasColourName":
            G.add_node(obj_filtered, label="ColourName")
            G.add_edge(subj_filtered, obj_filtered, relation="hasColourName")
        elif pred_filtered == "isOfGroup":
            G.add_node(obj_filtered, label="ColourGroup")
            G.add_edge(subj_filtered, obj_filtered, relation="isOfGroup")

        # 处理颜色定义的属性，例如亮度、饱和度等
        elif pred_filtered in ["brightness", "hue", "saturation", "temperature", "hex_string"]:
            G.add_node(obj_filtered, label="Attribute")
            G.add_edge(subj_filtered, obj_filtered, relation=pred_filtered)

        # 处理颜色组之间的关系
        elif pred_filtered in ["complementaryOf", "analogousWith", "triadicWith", "tetradicWith"]:
            G.add_node(subj_filtered, label="ColourGroup")
            G.add_node(obj_filtered, label="ColourGroup")
            G.add_edge(subj_filtered, obj_filtered, relation=pred_filtered)

        # 处理未明确指定的其他关系
        else:
            G.add_edge(subj_filtered, obj_filtered, relation=pred_filtered)

    # print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")
    # 获取所有实体（节点）的列表
    all_entities = list(G.nodes())
    # 给定的实体
    # 为每个实体构建 one-hot 向量
    node_features = {entity: [1 if i == entity_index else 0 for i in range(len(all_entities))] for
                     entity_index, entity in enumerate(all_entities)}

    return G, all_entities, node_features
    # 调用可视化函数
    visualize_graph(G)


# 可视化知识图谱
def visualize_graph(G):
    # 假设 G 是你的网络图，已经加载了节点和关系
    pos = nx.spring_layout(G, k=0.3, iterations=50)  # 布局设置

    plt.figure(figsize=(20, 20))  # 设置较大的图像尺寸

    # 绘制节点
    node_labels = {node: node for node, data in G.nodes(data=True)}  # 只显示节点名称
    node_sizes = [len(list(G.neighbors(node))) * 50 for node in G.nodes()]  # 根据节点度数调整节点大小
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # 绘制边并启用箭头
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, edge_color='black')  # arrows=True 启用 FancyArrowPatches

    # 绘制边上的标签（显示边的关系类型）
    edge_labels = nx.get_edge_attributes(G, 'relation')  # 获取边的关系标签
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=6)

    # 绘制图形
    plt.title("Colour Ontology Knowledge Graph")
    plt.show()


if __name__ == '__main__':
    get_graph_data()
=======
import rdflib
import networkx as nx
import matplotlib.pyplot as plt
import re


def get_graph_data():
    # 载入颜色知识图谱
    g = rdflib.Graph()
    g.parse("ontology.ttl", format="ttl")

    # 创建 NetworkX 图来表示知识图谱
    G = nx.Graph()

    # 解析类别文件
    category_file = "list_category_cloth.txt"
    with open(category_file, "r") as f:
        lines = f.readlines()
        category_count = int(lines[0].strip())  # 获取类别数量
        categories = [line.split() for line in lines[2:]]  # 从第三行开始读取

    # 解析属性文件
    attribute_file = "list_attr_cloth1.txt"
    with open(attribute_file, "r") as f:
        lines = f.readlines()
        attribute_count = int(lines[0].strip())  # 获取属性数量
        attributes = [re.split(r'\s{2,}', line.strip()) for line in lines[2:]]  # 从第三行开始读取

    # 将服装类别加入知识图谱
    for category_name, category_type in categories:
        G.add_node(category_name, type="Clothing Category", category_type=category_type)

    # 将服装属性加入知识图谱
    for attribute_name, attribute_type in attributes:
        G.add_node(attribute_name, type="Clothing Attribute", attribute_type=attribute_type)

    # 定义更加全面的连接规则
    for category_name, category_type in categories:
        for attribute_name, attribute_type in attributes:
            # 上半身衣物规则
            if category_type == "1":  # 上半身衣物
                if attribute_type == "1":  # 材质
                    G.add_edge(category_name, attribute_name, relation="has_texture")
                elif attribute_type == "2":  # 面料
                    G.add_edge(category_name, attribute_name, relation="has_fabric")
                elif attribute_type == "3":  # 形状
                    G.add_edge(category_name, attribute_name, relation="has_shape")
                elif attribute_type == "4":  # 部位
                    G.add_edge(category_name, attribute_name, relation="has_part")
                elif attribute_type == "5":  # 风格
                    G.add_edge(category_name, attribute_name, relation="has_style")

            # 下半身衣物规则
            elif category_type == "2":  # 下半身衣物
                if attribute_type == "1":  # 材质
                    G.add_edge(category_name, attribute_name, relation="has_texture")
                elif attribute_type == "2":  # 面料
                    G.add_edge(category_name, attribute_name, relation="has_fabric")
                elif attribute_type == "3":  # 形状
                    G.add_edge(category_name, attribute_name, relation="has_shape")
                elif attribute_type == "4":  # 部位
                    G.add_edge(category_name, attribute_name, relation="has_part")
                elif attribute_type == "5":  # 风格
                    G.add_edge(category_name, attribute_name, relation="has_style")

            # 全身衣物规则
            elif category_type == "3":  # 全身衣物
                if attribute_type == "1":  # 材质
                    G.add_edge(category_name, attribute_name, relation="has_texture")
                elif attribute_type == "2":  # 面料
                    G.add_edge(category_name, attribute_name, relation="has_fabric")
                elif attribute_type == "3":  # 形状
                    G.add_edge(category_name, attribute_name, relation="has_shape")
                elif attribute_type == "4":  # 部位
                    G.add_edge(category_name, attribute_name, relation="has_part")
                elif attribute_type == "5":  # 风格
                    G.add_edge(category_name, attribute_name, relation="has_style")

    # 整合颜色知识图谱（从颜色本体中获取颜色节点，并与服装类别、属性连接）
    namespace_manager = g.namespace_manager

    def remove_prefix(uri):
        uri_str = str(uri)
        for prefix, namespace in namespace_manager.namespaces():
            if uri_str.startswith(str(namespace)):
                return uri_str[len(str(namespace)):]
        return uri_str

    for subj, pred, obj in g:
        subj_filtered = remove_prefix(subj)
        pred_filtered = remove_prefix(pred)
        obj_filtered = remove_prefix(obj)

        # 处理 hasColourName 和 isOfGroup 关系
        if pred_filtered == "hasColourName":
            G.add_node(obj_filtered, label="ColourName")
            G.add_edge(subj_filtered, obj_filtered, relation="hasColourName")
        elif pred_filtered == "isOfGroup":
            G.add_node(obj_filtered, label="ColourGroup")
            G.add_edge(subj_filtered, obj_filtered, relation="isOfGroup")

        # 处理颜色定义的属性，例如亮度、饱和度等
        elif pred_filtered in ["brightness", "hue", "saturation", "temperature", "hex_string"]:
            G.add_node(obj_filtered, label="Attribute")
            G.add_edge(subj_filtered, obj_filtered, relation=pred_filtered)

        # 处理颜色组之间的关系
        elif pred_filtered in ["complementaryOf", "analogousWith", "triadicWith", "tetradicWith"]:
            G.add_node(subj_filtered, label="ColourGroup")
            G.add_node(obj_filtered, label="ColourGroup")
            G.add_edge(subj_filtered, obj_filtered, relation=pred_filtered)

        # 处理未明确指定的其他关系
        else:
            G.add_edge(subj_filtered, obj_filtered, relation=pred_filtered)

    # print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")
    # 获取所有实体（节点）的列表
    all_entities = list(G.nodes())
    # 给定的实体
    # 为每个实体构建 one-hot 向量
    node_features = {entity: [1 if i == entity_index else 0 for i in range(len(all_entities))] for
                     entity_index, entity in enumerate(all_entities)}

    return G, all_entities, node_features
    # 调用可视化函数
    visualize_graph(G)


# 可视化知识图谱
def visualize_graph(G):
    # 假设 G 是你的网络图，已经加载了节点和关系
    pos = nx.spring_layout(G, k=0.3, iterations=50)  # 布局设置

    plt.figure(figsize=(20, 20))  # 设置较大的图像尺寸

    # 绘制节点
    node_labels = {node: node for node, data in G.nodes(data=True)}  # 只显示节点名称
    node_sizes = [len(list(G.neighbors(node))) * 50 for node in G.nodes()]  # 根据节点度数调整节点大小
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # 绘制边并启用箭头
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, edge_color='black')  # arrows=True 启用 FancyArrowPatches

    # 绘制边上的标签（显示边的关系类型）
    edge_labels = nx.get_edge_attributes(G, 'relation')  # 获取边的关系标签
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=6)

    # 绘制图形
    plt.title("Colour Ontology Knowledge Graph")
    plt.show()


if __name__ == '__main__':
    get_graph_data()
>>>>>>> origin/main
