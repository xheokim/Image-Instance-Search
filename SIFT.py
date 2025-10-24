import os
import cv2
import numpy as np
import pickle
import re
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# %%
class SIFTInstanceSearch:
    def __init__(self, n_clusters=1000):
        self.n_clusters = n_clusters

        # 初始化SIFT检测器
        try:
            self.sift = cv2.SIFT_create()
            print("SIFT检测器初始化成功")
        except Exception as e:
            print(f"SIFT初始化失败: {e}")
            raise

        self.vocabulary = None
        self.kmeans = None

    def extract_sift_features(self, image_path):
        """提取单张图像的SIFT特征"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                return None

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)

            if descriptors is None:
                return None

            return descriptors
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            return None

    def extract_sift_features_from_image(self, image):
        """从图像数组提取SIFT特征"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            return descriptors
        except Exception as e:
            print(f"提取特征时出错: {e}")
            return None

    def extract_all_gallery_features(self, gallery_path):
        """提取所有gallery图像的SIFT特征"""
        print("提取gallery图像SIFT特征...")

        all_descriptors = []
        image_files = []

        # 获取所有图像文件
        files = [f for f in os.listdir(gallery_path)
                 if f.lower().endswith(('png', 'jpg', 'jpeg'))]

        print(f"找到 {len(files)} 张图像")

        for filename in tqdm(files, desc="提取SIFT特征"):
            image_path = os.path.join(gallery_path, filename)
            descriptors = self.extract_sift_features(image_path)

            if descriptors is not None and len(descriptors) > 0:
                all_descriptors.extend(descriptors)
                image_files.append(filename)

        print(f"特征提取完成，共 {len(all_descriptors)} 个描述符，来自 {len(image_files)} 张图像")
        return all_descriptors, image_files

    def build_visual_vocabulary(self, all_descriptors):
        """构建视觉词典"""
        print("构建视觉词典...")

        # 如果描述符太多，进行采样以加速训练
        if len(all_descriptors) > 100000:
            print(f"描述符数量过多 ({len(all_descriptors)})，进行采样到100000...")
            indices = np.random.choice(len(all_descriptors), 100000, replace=False)
            all_descriptors_sampled = [all_descriptors[i] for i in indices]
        else:
            all_descriptors_sampled = all_descriptors

        # 转换为numpy数组
        descriptors_array = np.array(all_descriptors_sampled)
        print(f"训练数据形状: {descriptors_array.shape}")

        # 使用K-means聚类
        print("开始K-means聚类...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10, verbose=0)
        self.kmeans.fit(descriptors_array)
        self.vocabulary = self.kmeans.cluster_centers_

        print(f"视觉词典构建完成，词汇量: {self.n_clusters}")
        return self.vocabulary

    def image_to_bow(self, descriptors):
        """将图像描述符转换为词袋向量"""
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_clusters)

        # 为每个描述符找到最近的视觉单词
        labels = self.kmeans.predict(descriptors)

        # 构建词袋直方图
        bow_vector = np.zeros(self.n_clusters)
        for label in labels:
            bow_vector[label] += 1

        # 归一化
        if np.sum(bow_vector) > 0:
            bow_vector = bow_vector / np.sum(bow_vector)

        return bow_vector

    def process_gallery_images(self, gallery_path, force_retrain=False):
        """处理所有gallery图像并构建词袋特征数据库"""

        vocabulary_path = "sift_vocabulary.pkl"
        bow_features_path = "sift_bow_features.pkl"

        # 检查是否已有训练好的词典和特征
        if not force_retrain and os.path.exists(vocabulary_path) and os.path.exists(bow_features_path):
            print("加载已训练的视觉词典和特征...")
            try:
                with open(vocabulary_path, 'rb') as f:
                    self.vocabulary, self.kmeans = pickle.load(f)
                with open(bow_features_path, 'rb') as f:
                    bow_features_dict = pickle.load(f)
                print(f"加载成功: {len(bow_features_dict)} 张图像的特征")
                return bow_features_dict
            except Exception as e:
                print(f"加载失败: {e}，重新训练...")
                force_retrain = True

        if force_retrain or not os.path.exists(vocabulary_path) or not os.path.exists(bow_features_path):
            # 提取所有特征
            all_descriptors, image_files = self.extract_all_gallery_features(gallery_path)

            if len(all_descriptors) == 0:
                print("错误: 没有提取到任何特征！")
                return {}

            # 构建视觉词典
            self.build_visual_vocabulary(all_descriptors)

            # 为每张图像构建词袋特征
            print("为gallery图像构建词袋特征...")
            bow_features_dict = {}

            successful_count = 0
            for filename in tqdm(image_files, desc="构建词袋特征"):
                image_path = os.path.join(gallery_path, filename)
                descriptors = self.extract_sift_features(image_path)

                if descriptors is not None and len(descriptors) > 0:
                    bow_vector = self.image_to_bow(descriptors)
                    bow_features_dict[filename] = bow_vector
                    successful_count += 1

            print(f"成功为 {successful_count}/{len(image_files)} 张图像构建特征")

            # 保存词典和特征
            try:
                with open(vocabulary_path, 'wb') as f:
                    pickle.dump((self.vocabulary, self.kmeans), f)
                with open(bow_features_path, 'wb') as f:
                    pickle.dump(bow_features_dict, f)
                print("特征保存成功")
            except Exception as e:
                print(f"保存特征时出错: {e}")

        return bow_features_dict

    def search_similar_images(self, query_bow_vector, bow_features_dict, top_k=10):
        """搜索最相似的图像"""
        similarities = {}

        # 准备特征矩阵
        gallery_filenames = list(bow_features_dict.keys())
        gallery_features = np.array([bow_features_dict[name] for name in gallery_filenames])

        # 计算余弦相似度
        query_features = query_bow_vector.reshape(1, -1)
        sim_scores = cosine_similarity(query_features, gallery_features)[0]

        # 获取最相似的图像
        for i, filename in enumerate(gallery_filenames):
            similarities[filename] = sim_scores[i]

        # 按相似度排序
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        return sorted_similarities[:top_k]

    def search_all_similar_images(self, query_bow_vector, bow_features_dict):
        """搜索所有相似图像（返回完整排序列表）"""
        similarities = {}

        # 准备特征矩阵
        gallery_filenames = list(bow_features_dict.keys())
        gallery_features = np.array([bow_features_dict[name] for name in gallery_filenames])

        # 计算余弦相似度
        query_features = query_bow_vector.reshape(1, -1)
        sim_scores = cosine_similarity(query_features, gallery_features)[0]

        # 获取所有图像的相似度
        for i, filename in enumerate(gallery_filenames):
            similarities[filename] = sim_scores[i]

        # 按相似度排序（降序）
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        return sorted_similarities

    def load_bboxes(self, txt_path):
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

    def bbox_to_xyxy(self, bbox, img_width, img_height):
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

    def extract_features_from_bbox(self, image_path, bbox):
        """从边界框区域提取特征"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                # 确保边界框有效
                if x2 > x1 and y2 > y1:
                    cropped_image = image[y1:y2, x1:x2]
                    if cropped_image.size == 0:
                        print("边界框区域为空，使用整张图像")
                        cropped_image = image
                else:
                    print("边界框无效，使用整张图像")
                    cropped_image = image
            else:
                cropped_image = image

            return self.extract_sift_features_from_image(cropped_image)

        except Exception as e:
            print(f"从边界框提取特征时出错: {e}")
            return None


# %%
def generate_rank_list_file(all_query_results, output_file="rankList_sift.txt"):
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
                # 使用正则表达式提取所有数字
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
def main_sift_ranklist():
    """SIFT主函数 - 生成排名列表"""

    # 初始化SIFT实例搜索
    print(" 初始化SIFT实例搜索系统...")
    sift_searcher = SIFTInstanceSearch(n_clusters=1000)

    # 处理gallery图像并构建特征数据库
    gallery_path = "gallery_images"
    if not os.path.exists(gallery_path):
        print(f"错误: 图库目录 {gallery_path} 不存在")
        return

    print("构建特征数据库...")
    bow_features_dict = sift_searcher.process_gallery_images(gallery_path)

    if not bow_features_dict:
        print("错误: 无法构建特征数据库")
        return

    print(f"特征数据库包含 {len(bow_features_dict)} 张图像")

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

    # 存储所有查询结果
    all_query_results = {}

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
        bboxes = sift_searcher.load_bboxes(txt_path)
        if not bboxes:
            print(f"查询 {i} 没有找到边界框，使用整张图像")
            bbox = None
        else:
            bbox = sift_searcher.bbox_to_xyxy(bboxes[0], img_width, img_height)
            print(f"使用边界框: {bbox}")

        # 从边界框区域提取SIFT特征
        query_descriptors = sift_searcher.extract_features_from_bbox(query_image_path, bbox)

        if query_descriptors is None or len(query_descriptors) == 0:
            print(f"无法从查询 {i} 提取特征")
            all_query_results[i] = []  # 空结果
            continue

        print(f"查询特征数量: {len(query_descriptors)} 个描述符")

        # 转换为词袋向量
        query_bow_vector = sift_searcher.image_to_bow(query_descriptors)

        # 搜索所有相似图像（返回完整排序列表）
        print(f"为查询 {i} 搜索相似图像...")
        results = sift_searcher.search_all_similar_images(query_bow_vector, bow_features_dict)

        # 存储这个查询的所有结果
        all_query_results[i] = results

        # 显示前5个结果
        print(f"查询 {i} 的前5个结果:")
        for rank, (filename, similarity) in enumerate(results[:5], 1):
            print(f"  Rank {rank}: {filename} (相似度: {similarity:.4f})")

    # 生成排名列表文件
    generate_rank_list_file(all_query_results, "rankList_sift.txt")

    print("\n SIFT排名列表生成完成！")
    print("文件已保存为: rankList_sift.txt")


# %%
# 运行主函数
if __name__ == "__main__":
    main_sift_ranklist()