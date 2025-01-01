import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from itertools import combinations
from scipy.stats import gaussian_kde
from itertools import combinations
from PIL import Image, ImageDraw
import io
import cv2

# WSFCM 聚类算法类
class WSFCM:
    def __init__(self, n_clusters=2, max_iter=300, m=2.0, epsilon=1e-10, a=1, c=0, d=1, beta=2.0, lambda_sparse=0.1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.epsilon = epsilon
        # self.gamma = gamma
        self.a = a
        self.c = c
        self.d = d
        self.beta = beta
        self.lambda_sparse = lambda_sparse
        self.feature_weights = None  # 初始化特征权重为空，稍后初始化

    def initialize_membership_matrix(self, n_samples):
        membership_mat = np.random.rand(n_samples, self.n_clusters)
        membership_mat = membership_mat / np.sum(membership_mat, axis=1, keepdims=True)
        return membership_mat
    
    def initialize_feature_weights(self, n_features):
        # 随机选择一个特征赋权重值为1，其他特征权重为0
        self.feature_weights = np.zeros(n_features)
        self.feature_weights[np.random.randint(0, n_features)] = 1

    def compute_cluster_centers(self, X, membership_mat):
        cluster_centers = []
        for j in range(self.n_clusters):
            numerator = np.sum((membership_mat[:, j]**self.m).reshape(-1, 1) * X, axis=0)
            denominator = np.sum(membership_mat[:, j]**self.m)
            cluster_centers.append(numerator / denominator)
        return np.array(cluster_centers)

    def update_feature_weights(self, X, membership_mat, cluster_centers):
        n_features = X.shape[1]
        D = np.zeros(n_features)
        for j in range(n_features):
            D[j] = np.sum([np.sum([membership_mat[i, h]**self.m * (self.euclidean_distance(X[i, j], cluster_centers[h, j]))
                                   for i in range(X.shape[0])]) for h in range(self.n_clusters)])
            D[j] += self.lambda_sparse * np.sum([np.sum([membership_mat[i, h]**self.m * abs(X[i, j])
                                                        for i in range(X.shape[0])]) for h in range(self.n_clusters)])
        D_nonzero = D[D != 0]
        for j in range(n_features):
            if D[j] != 0:
                self.feature_weights[j] = 1 / np.sum([(D[j] / D_t) ** (1 / (self.beta - 1)) for D_t in D_nonzero])
    
    def update_membership_matrix(self, X, cluster_centers):
        membership_mat = np.zeros((X.shape[0], self.n_clusters))
        for i in range(X.shape[0]):
            for h in range(self.n_clusters):
                B_ih = np.sum([self.feature_weights[j]**self.beta * (self.euclidean_distance(X[i, j], cluster_centers[h, j]) + self.lambda_sparse * abs(X[i, j]))
                               for j in range(X.shape[1])])
                denominator = np.sum([(B_ih / B_is) ** (1 / (self.m - 1)) for B_is in [np.sum([self.feature_weights[j]**self.beta * (self.euclidean_distance(X[i, j], cluster_centers[s, j]) + self.lambda_sparse * abs(X[i, j]))
                                                                                            for j in range(X.shape[1])]) for s in range(self.n_clusters)]])
                membership_mat[i, h] = 1 / denominator
        return membership_mat
    
    def euclidean_distance(self, point1, point2):
        #distance = (point1 - point2) ** 2
        #return 2-2*(np.exp(-gamma * distance))
        #return distance
        return (self.a*point1**2 + self.c)**self.d - 2*(self.a*point1*point2 + self.c)**self.d + (self.a*point2**2 + self.c)**self.d 
        #return np.tanh(self.a*point1**2 + self.c) - 2*np.tanh(self.a*point1*point2 + self.c) + np.tanh(self.a*point2**2 + self.c)**self.d 
    
    def fit(self, X):
        self.initialize_feature_weights(X.shape[1])
        membership_mat = self.initialize_membership_matrix(X.shape[0])
        for iteration in range(self.max_iter):
            cluster_centers = self.compute_cluster_centers(X, membership_mat)
            self.update_feature_weights(X, membership_mat, cluster_centers)
            new_membership_mat = self.update_membership_matrix(X, cluster_centers)
            if np.linalg.norm(new_membership_mat - membership_mat) < self.epsilon:
                break
            membership_mat = new_membership_mat
        self.cluster_centers_ = cluster_centers
        self.labels_ = membership_mat.argmax(axis=1)
        self.membership_mat_ = membership_mat

    def predict(self, X):
        if hasattr(self, 'cluster_centers_'):
            membership_mat = self.update_membership_matrix(X, self.cluster_centers_)
            return membership_mat.argmax(axis=1)
        else:
            raise Exception("Model not fitted, please run 'fit' method first.")

# Streamlit 应用标题
st.title("自动数据标注工具")

# 数据标注部分
uploaded_file = st.file_uploader("上传数据文件 (CSV 或 Excel 格式)", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        data = pd.read_excel(uploaded_file)
    st.write("原始数据：")
    st.dataframe(data)

    st.sidebar.subheader("特征选择")
    selected_features = st.sidebar.multiselect("选择用于聚类的特征", options=data.columns.tolist(), default=data.columns.tolist())

    # 高级参数设置
    if st.sidebar.checkbox("显示高级参数设置"):
        max_iter = st.sidebar.slider("最大迭代次数", min_value=100, max_value=1000, value=300, step=50)
        m = st.sidebar.slider("模糊系数 (m)", min_value=1.1, max_value=3.0, value=2.0, step=0.1)
        beta = st.sidebar.slider("权重系数 (beta)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
        lambda_sparse = st.sidebar.slider("稀疏系数 (lambda)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        a = st.sidebar.slider("距离参数 a", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        c = st.sidebar.slider("距离参数 c", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
        d = st.sidebar.slider("距离参数 d", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    else:
        max_iter = 300
        m = 2.0
        beta = 2.0
        lambda_sparse = 0.0
        a = 1.0
        c = 0.0
        d = 1.0

    n_clusters = st.sidebar.number_input("聚类数目 (n_clusters)", min_value=2, max_value=20, value=3, step=1)

    if selected_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(data[selected_features])

        # 应用 WSFCM 聚类
        st.write("正在进行聚类分析...")
        wsfcm = WSFCM(n_clusters=n_clusters, max_iter=max_iter, m=m, beta=beta, lambda_sparse=lambda_sparse, a=a, c=c, d=d)
        wsfcm.fit(X)
        labels = wsfcm.labels_
        membership_mat = wsfcm.membership_mat_
        data = data[selected_features].copy()
        data['Cluster'] = labels

        # 显示特征权重
        feature_weights = wsfcm.feature_weights  # 假设核心算法输出特征权重
        feature_weight_df = pd.DataFrame({"Feature": selected_features, "Weight": feature_weights})
        st.write("特征权重：")
        st.dataframe(feature_weight_df)

        # 高置信度样本标注
        confidence_threshold = st.sidebar.slider("高置信度阈值", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
        high_confidence = membership_mat.max(axis=1) >= confidence_threshold
        high_confidence_samples = data[high_confidence]
        st.write("高置信度样本：")
        st.dataframe(high_confidence_samples)

        # 标注样本训练分类器
        st.write("训练分类器...")
        X_train, X_test, y_train, y_test = train_test_split(
            high_confidence_samples[selected_features], high_confidence_samples['Cluster'], test_size=0.3, random_state=42
        )
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)

        # 迭代优化
        st.write("迭代优化中...")
        for iteration in range(5):  # 限制迭代次数
            remaining_samples = data[~high_confidence]
            if remaining_samples.empty:
                break
            predicted_labels = classifier.predict(remaining_samples[selected_features])
            predicted_confidences = classifier.predict_proba(remaining_samples[selected_features]).max(axis=1)
            high_confidence_new = predicted_confidences >= confidence_threshold
            new_samples = remaining_samples[high_confidence_new]
            high_confidence_samples = pd.concat([high_confidence_samples, new_samples])
            classifier.fit(high_confidence_samples[selected_features], high_confidence_samples['Cluster'])

        # 下载标注结果
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="下载标注数据",
            data=csv,
            file_name="annotated_data.csv",
            mime="text/csv"
        )

        # 标注结果可视化
        st.write("标注结果可视化：")
        feature_pairs = list(combinations(selected_features, 2))
        fig, axes = plt.subplots(len(feature_pairs), 1, figsize=(10, 5 * len(feature_pairs)))
        if len(feature_pairs) == 1:
            axes = [axes]
        for ax, (x_feature, y_feature) in zip(axes, feature_pairs):
            for cluster in range(n_clusters):
                cluster_data = data[data['Cluster'] == cluster]
                ax.scatter(cluster_data[x_feature], cluster_data[y_feature], label=f'Cluster {cluster}', alpha=0.6)
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.legend()
        st.pyplot(fig)

# 图片标注部分
st.sidebar.subheader("图片标注")
uploaded_images = st.file_uploader("上传图片文件 (支持 JPG/PNG)", type=["jpg", "png"], accept_multiple_files=True)

def extract_image_features(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    edges = cv2.Canny(gray, 100, 200)
    edge_hist = np.histogram(edges, bins=16, range=(0, 256))[0]
    edge_hist = edge_hist / np.sum(edge_hist)
    features = np.concatenate([hist, edge_hist])
    return features

if uploaded_images:
    # 图片高级参数设置
    if st.sidebar.checkbox("显示图片标注高级参数设置"):
        img_max_iter = st.sidebar.slider("最大迭代次数", min_value=100, max_value=1000, value=300, step=50)
        img_m = st.sidebar.slider("模糊系数 (m)", min_value=1.1, max_value=3.0, value=2.0, step=0.1)
        img_beta = st.sidebar.slider("权重系数 (beta)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
        img_lambda_sparse = st.sidebar.slider("稀疏系数 (lambda)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        img_a = st.sidebar.slider("距离参数 a", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        img_c = st.sidebar.slider("距离参数 c", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
        img_d = st.sidebar.slider("距离参数 d", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    else:
        img_max_iter = 300
        img_m = 2.0
        img_beta = 2.0
        img_lambda_sparse = 0.0
        img_a = 1.0
        img_c = 0.0
        img_d = 1.0

    n_clusters = st.sidebar.number_input("聚类数目 (n_clusters)", min_value=2, max_value=20, value=3, step=1)
    
    st.write("已上传图片：")
    image_features = []
    images = [Image.open(img) for img in uploaded_images]

    for img in images:
        st.image(img, caption="原始图片", use_container_width=True)
        features = extract_image_features(img)
        image_features.append(features)

    image_features = np.array(image_features)

    # 使用 WSFCM 聚类图片
    st.write("正在进行图片聚类...")
    wsfcm = WSFCM(n_clusters=n_clusters, max_iter=img_max_iter, m=img_m, beta=img_beta, lambda_sparse=img_lambda_sparse, a=img_a, c=img_c, d=img_d)
    wsfcm.fit(image_features)
    image_labels = wsfcm.labels_

    # 标注图片
    labeled_images = []
    for i, (img, label) in enumerate(zip(images, image_labels)):
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, img.width-10, img.height-10], outline="red", width=5)
        draw.text((20, 20), f"Cluster {label}", fill="red")
        labeled_images.append(img)

    st.write("标注结果：")
    for labeled_img in labeled_images:
        st.image(labeled_img, caption="标注图片", use_container_width=True)

    st.write("下载标注图片：")
    for i, labeled_img in enumerate(labeled_images):
        buffer = io.BytesIO()
        labeled_img.save(buffer, format="PNG")
        buffer.seek(0)
        st.download_button(
            label=f"下载图片 {i+1}",
            data=buffer,
            file_name=f"labeled_image_{i+1}.png",
            mime="image/png"
        )