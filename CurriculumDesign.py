from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# 读取数据集
wine = pd.read_csv("wine.csv")
wine_quality = pd.read_csv("wine_quality.csv", sep=';')

# 将数据集的数据和标签拆分开
wine = wine.iloc[:, 1:]
wine_labels = wine.iloc[:, 0]
wine_data = wine.values
wine_quality = wine_quality.iloc[:, :-1]
wine_quality_labels = wine_quality.iloc[:, -1]
wine_quality_data = wine_quality.values
print("\nwine数据集的数据：")
print(wine_data)
print("\nwine_quality数据集的数据：")
print(wine_quality_data)
print("\nwine数据集的标签")
print(wine_labels)
print("\nwine_quality数据集的标签")
print(wine_quality_labels)

# 将数据集划分为训练集和测试集 测试集比例为10%  (test_size表示测试集对数据集的占比  random_state确保实验结果可复现)
wine_data_train, wine_data_test, wine_labels_train, wine_labels_test = train_test_split(wine_data, wine_labels,
                                                                                        test_size=0.1, random_state=42)
wine_quality_data_train, wine_quality_data_test, wine_quality_labels_train, wine_quality_labels_test = train_test_split(
    wine_quality_data, wine_quality_labels, test_size=0.1, random_state=42)
print("\nwine数据集的训练集：")
print(wine_data_train)
print("\nwine_quality数据集的训练集：")
print(wine_quality_data_train)

# 标准化数据集 对wine数据集采用均值标准化
wine_data_train_df = pd.DataFrame(wine_data_train)
scaler = StandardScaler()
wine_train_standardized = scaler.fit_transform(wine_data_train_df)
wine_data_test_df = pd.DataFrame(wine_data_test)
scaler = StandardScaler()
wine_test_standardized = scaler.fit_transform(wine_data_test_df)
print("\nwine数据训练集的均值标准化结果：")
print(wine_train_standardized)
print("\nwine数据测试集的均值标准化结果：")
print(wine_test_standardized)

# 标准化数据集 对wine_quality数据集采用均值标准化
wine_quality_data_test_df = pd.DataFrame(wine_quality_data_test)
scaler = StandardScaler()
wine_quality_test_standardized = scaler.fit_transform(wine_quality_data_test_df)
wine_quality_data_train_df = pd.DataFrame(wine_quality_data_train)
scaler = StandardScaler()
wine_quality_train_standardized = scaler.fit_transform(wine_quality_data_train_df)
print("\nwine_quality数据训练集的均值标准化结果：")
print(wine_quality_train_standardized)
print("\nwine_quality数据测试集的均值标准化结果：")
print(wine_quality_test_standardized)

# PCA降维 n_components表示保留前五个方差最大的特征向量
# 对wine数据集进行PCA降维
pca = PCA(n_components=5).fit(wine_train_standardized)
wine_trainPCA = pca.transform(wine_train_standardized)
wine_testPCA = pca.transform(wine_test_standardized)
print("\nwine数据训练集的PCA降维结果：")
print(wine_trainPCA)
print("\nwine数据测试集的PCA降维结果：")
print(wine_testPCA)
# 对wine_quality数据集进行PCA降维
pca = PCA(n_components=5).fit(wine_quality_train_standardized)
wine_quality_trainPCA = pca.transform(wine_quality_train_standardized)
wine_quality_testPCA = pca.transform(wine_quality_test_standardized)
print("\nwine_quality数据训练集的PCA降维结果：")
print(wine_quality_trainPCA)
print("\nwine_quality数据测试集的PCA降维结果：")
print(wine_quality_testPCA)
