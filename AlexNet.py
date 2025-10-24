#导入对应的函数库
import os
import pickle
import numpy as np
import re
from PIL import Image

# PyTorch相关import
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

# sklearn相关import
from sklearn.metrics.pairwise import cosine_similarity


def extract_alexnet_features_from_image(image_path, bbox=None):
    """使用预训练的AlexNet提取特征（简化版本）"""
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image

    # 加载预训练的AlexNet
    model = models.alexnet(pretrained=True)
    # 移除最后的分类层，使用特征提取部分
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')

        # 如果有边界框，裁剪区域
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            image = image.crop((x1, y1, x2, y2))

        # 预处理
        image_tensor = transform(image).unsqueeze(0)

        # 提取特征
        with torch.no_grad():
            features = feature_extractor(image_tensor)
            features = features.view(features.size(0), -1)

        return features.numpy().flatten()

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


# %%
def load_bboxes(txt_path):
    """加载边界框信息"""
    bboxes = []
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                data = line.strip().split()
                if len(data) >= 5:
                    bboxes.append([float(x) for x in data[1:5]])
        return bboxes
    except:
        return []


# %%
def bbox_to_xyxy(bbox, img_width, img_height):
    """将YOLO格式转换为实际坐标"""
    x_center, y_center, width, height = bbox
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)

    # 确保坐标在图像范围内
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))

    return [x1, y1, x2, y2]


# %%
def extract_all_gallery_features(gallery_path, feature_extractor):
    """提取所有gallery图像的特征"""
    features_dict = {}

    print("开始提取gallery图像特征...")
    image_files = [f for f in os.listdir(gallery_path)
                   if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for i, filename in enumerate(image_files):
        if i % 1000 == 0:
            print(f"已处理 {i}/{len(image_files)} 张图像")

        image_path = os.path.join(gallery_path, filename)
        features = feature_extractor.extract_features(image_path)

        if features is not None:
            features_dict[filename] = features

    print(f"特征提取完成，共提取 {len(features_dict)} 张图像的特征")
    return features_dict


# %%
def search_similar_images_all(query_features, gallery_features_dict):
    """搜索所有相似图像（返回完整排序列表）"""
    similarities = {}

    # 准备特征矩阵
    gallery_filenames = list(gallery_features_dict.keys())
    gallery_features = np.array([gallery_features_dict[name] for name in gallery_filenames])

    # 计算余弦相似度
    query_features = query_features.reshape(1, -1)
    sim_scores = cosine_similarity(query_features, gallery_features)[0]

    # 获取所有图像的相似度
    for i, filename in enumerate(gallery_filenames):
        similarities[filename] = sim_scores[i]

    # 按相似度排序（降序）
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    return sorted_similarities


# %%
def visualize_results(query_image_path, bbox, results, gallery_path, query_id):
    """可视化查询结果"""
    # 加载查询图像并绘制边界框
    query_img = Image.open(query_image_path).convert('RGB')
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        query_img_with_bbox = query_img.copy()
        draw = ImageDraw.Draw(query_img_with_bbox)
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
    else:
        query_img_with_bbox = query_img

    # 创建结果图
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    axes = axes.ravel()

    # 显示查询图像
    axes[0].imshow(query_img_with_bbox)
    axes[0].set_title(f'Query Image {query_id}')
    axes[0].axis('off')

    # 显示前11个结果（跳过第一个位置）
    for i, (result_filename, similarity) in enumerate(results[:11]):
        result_path = os.path.join(gallery_path, result_filename)
        try:
            result_img = Image.open(result_path).convert('RGB')
            axes[i + 1].imshow(result_img)
            axes[i + 1].set_title(f'Rank {i + 1}\nSim: {similarity:.3f}')
            axes[i + 1].axis('off')
        except:
            axes[i + 1].text(0.5, 0.5, f'Error loading\n{result_filename}',
                             ha='center', va='center')
            axes[i + 1].axis('off')

    # 隐藏多余的子图
    for i in range(len(results) + 1, 12):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'alexnet_query_{query_id}_results.png', dpi=150, bbox_inches='tight')
    plt.show()


# %%
def safe_save_features(features_dict, filepath):
    """安全保存特征"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(features_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f" 特征保存成功: {filepath}")
        return True
    except Exception as e:
        print(f" 特征保存失败: {e}")
        return False


def safe_load_features(filepath):
    """安全加载特征"""
    try:
        with open(filepath, 'rb') as f:
            features_dict = pickle.load(f)
        print(f" 特征加载成功: {len(features_dict)} 个特征")
        return features_dict
    except Exception as e:
        print(f" 特征加载失败: {e}")
        return None


# %%
def generate_rank_list_file(all_query_results, output_file="rankList_alexnet.txt"):
    """生成排名列表文件"""
    print(f"生成排名列表文件: {output_file}")

    with open(output_file, 'w') as f:
        for query_id in sorted(all_query_results.keys()):
            results = all_query_results[query_id]
            # 提取图像编号（从文件名中提取数字）
            image_numbers = []
            for filename, similarity in results:
                # 从文件名中提取数字
                base_name = os.path.splitext(filename)[0]
                # 移除可能的文件扩展名和非数字字符
                import re
                numbers = re.findall(r'\d+', base_name)
                if numbers:
                    image_numbers.append(numbers[0])  # 使用找到的第一个数字
                else:
                    image_numbers.append(base_name)  # 如果没有数字，使用原名称

            # 写入格式: Q1: 7 12 214 350...
            line = f"Q{query_id}: " + " ".join(image_numbers) + "\n"
            f.write(line)

    print(f"排名列表已保存到: {output_file}")
    print(f"总共处理了 {len(all_query_results)} 个查询")


# %%
def main_alexnet_ranklist():
    """AlexNet主函数 - 生成排名列表"""

    # 检查特征数据库文件是否存在
    gallery_features_path = "gallery_features_alexnet.pkl"
    if not os.path.exists(gallery_features_path):
        print(f"错误: 特征数据库文件 {gallery_features_path} 不存在")
        print("请确保文件在正确路径下")
        return

    # 加载gallery特征
    print("加载gallery特征数据库...")
    with open(gallery_features_path, 'rb') as f:
        gallery_features_dict = pickle.load(f)

    print(f"成功加载 {len(gallery_features_dict)} 个图像的特征")

    # 存储所有查询结果
    all_query_results = {}

    # 处理查询图像
    query_images_path = "query_images"
    query_txt_path = "query_txt"

    # 检查目录是否存在
    if not os.path.exists(query_images_path):
        print(f"错误: 查询图像目录 {query_images_path} 不存在")
        return

    if not os.path.exists(query_txt_path):
        print(f"错误: 查询文本目录 {query_txt_path} 不存在")
        return

    # 处理所有50个查询
    for i in range(1, 51):
        # 尝试不同的文件名格式
        possible_filenames = [
            f"query_{i}.jpg",
            f"query_{i}.png",
            f"query_{i}.jpeg",
            f"{i}.jpg",
            f"{i}.png"
        ]

        query_filename = None
        query_image_path = None

        # 查找实际存在的文件
        for filename in possible_filenames:
            temp_path = os.path.join(query_images_path, filename)
            if os.path.exists(temp_path):
                query_filename = filename
                query_image_path = temp_path
                break

        if query_image_path is None:
            print(f"跳过查询 {i}，未找到图像文件")
            all_query_results[i] = []  # 空结果
            continue

        # 对应的文本文件
        possible_txtnames = [
            f"query_{i}.txt",
            f"{i}.txt",
            query_filename.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        ]

        txt_path = None
        for txtname in possible_txtnames:
            temp_txt_path = os.path.join(query_txt_path, txtname)
            if os.path.exists(temp_txt_path):
                txt_path = temp_txt_path
                break

        if txt_path is None:
            print(f"跳过查询 {i}，边界框文件不存在")
            all_query_results[i] = []  # 空结果
            continue

        print(f"处理查询 {i}: {query_filename}")

        # 加载查询图像获取尺寸
        try:
            query_img = Image.open(query_image_path)
            img_width, img_height = query_img.size
        except Exception as e:
            print(f"加载查询图像 {query_image_path} 时出错: {e}")
            all_query_results[i] = []  # 空结果
            continue

        # 加载边界框
        bboxes = load_bboxes(txt_path)
        if not bboxes:
            print(f"查询 {i} 没有找到边界框，使用整张图像")
            bbox = None
        else:
            bbox = bbox_to_xyxy(bboxes[0], img_width, img_height)
            print(f"使用边界框: {bbox}")

        # 提取查询特征
        query_features = extract_alexnet_features_from_image(query_image_path, bbox)
        if query_features is None:
            print(f"无法提取查询 {i} 的特征")
            all_query_results[i] = []  # 空结果
            continue

        # 搜索相似图像（返回所有28493张图像的排序）
        print(f"为查询 {i} 搜索相似图像...")
        results = search_similar_images_all(query_features, gallery_features_dict)

        # 存储这个查询的所有结果
        all_query_results[i] = results

        # 显示前5个结果
        print(f"查询 {i} 的前5个结果:")
        for rank, (filename, similarity) in enumerate(results[:5], 1):
            print(f"  Rank {rank}: {filename} (相似度: {similarity:.4f})")

    # 生成排名列表文件
    generate_rank_list_file(all_query_results, "rankList_alexnet.txt")

    print("\n AlexNet排名列表生成完成！")


# 运行主函数
if __name__ == "__main__":
    main_alexnet_ranklist()