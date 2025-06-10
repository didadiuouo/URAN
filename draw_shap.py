import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from matplotlib.lines import Line2D
import shap
import pandas as pd
import numpy as np
def draw_shap(model,X_test_tensor,feature_names):

    # 设置全局字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体 (SimHei) 字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    # 假设你有一个特征名称列表
    feature_names = feature_names

    model.eval()
    # 使用SHAP的DeepExplainer进行特征重要性评估
    explainer = shap.GradientExplainer(model.cpu(), X_test_tensor)

    # 计算SHAP值
    shap_values = explainer.shap_values(X_test_tensor)

    # 检查 shap_values 的形状：对于多分类问题，shap_values 可能是一个列表
    # 选择某个类别进行可视化（假设二分类，选择第一个类别）
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # 选择第一个类别

    # 转换 X_test_ex 为 NumPy 格式以便用于 SHAP 绘图
    X_test_ex_np = X_test_tensor.detach().cpu().numpy()
    # 获取第二列及以后的数据
    feature_names = feature_names.iloc[:, 1:(X_test_ex_np.shape[1]+1)].T
    print(X_test_ex_np.shape[1], len(feature_names))

    # 确保 feature_names 与 X_test_ex 的列数量一致
    assert len(feature_names) == X_test_ex_np.shape[1], "特征名称与数据的特征数量不一致"

    feature_names = np.array(feature_names)

    # 绘制 SHAP 平均特征重要性条形图
    shap.summary_plot(shap_values[:, :, 0], X_test_ex_np, feature_names=feature_names)


    # 计算每个特征 SHAP 值的绝对值的平均值
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

    # 确保 mean_abs_shap_values 为一维
    if mean_abs_shap_values.ndim > 1:
        mean_abs_shap_values = mean_abs_shap_values[:, 0]

    # 将平均 SHAP 值与特征名称放入 DataFrame 中
    mean_abs_shap_df = pd.DataFrame(mean_abs_shap_values, index=feature_names, columns=['Mean |SHAP Value|'])

    # 对特征按平均 SHAP 值进行排序，并选择前20个重要特征
    mean_abs_shap_df = mean_abs_shap_df.sort_values(by='Mean |SHAP Value|', ascending=False).head(20)

    # 绘制水平条形图
    plt.figure(figsize=(10, 8))
    mean_abs_shap_df.plot(kind='barh', legend=False, color='skyblue')
    plt.title('Mean Absolute SHAP Values for Each Feature')
    plt.xlabel('Mean |SHAP Value|')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()  # 使最高重要性特征排在顶部
    plt.tight_layout()
    plt.show()
