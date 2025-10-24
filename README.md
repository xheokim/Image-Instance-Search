# Image-Instance-Search
基于本地下载下来的数据集中289493张图片进行训练。采用AlexNet深层卷积神经网络以及SIFT—词袋模型传统图像特征分析方法。实现“Show the top ten matching images with the bounding box(es) of instance(s) for queries 1-5”

1.文件包含zip格式的数据集，解压后包含训练集图片gallery_images，测试集图片query_images以及txt格式测试集集特征框文件query_txt。

2.文件gallery_features_alexnet.pkl，为AlexNet卷积深层神经网络的训练集特征提取数据库。（预训练模型：torchvision.models.alexnet(pretrained=True) 中的预训练模型是自动下载的！这是PyTorch官方提供的在ImageNet-1k数据集上预训练的AlexNet模型，模型文件约~230MB。高效简洁。）文件gallery_features_alexnet.pkl可以删去，供自行修改后的数据集实现重新提取新的特征数据库。

3.文件sift_vocabulary.pkl是SIFT算法下的视觉词典；文件sift_bow_features.pkl是图像的词袋特征；文件sift_query_X_results.png是查询结果可视化。三个文件也都可以删去，供自行修改后的数据集实现重新提取新的特征数据库。

4.VIassignment1_AlexNet.ipynb为AlexNet算法下的JupyterNotebook文件，包含算法的中文介绍与应用想法，包含源码。（AlexNet.py）

5.VIassignment1_SIFT.ipynb为SIFT算法下的JupyterNotebook文件，包含算法的中文介绍与应用想法，包含源码。（SIFT.py）

6.最终结果除了生成query1-5中每个query的前十个最相近的instance search图像外，还会将 50 个查询的检索结果（包含 28,493 幅图像的排序列表，按相似度降序排列）列于文本文件 rankList.txt 中。两种方式会各生成一个txt文件。
