import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from matplotlib.lines import Line2D
import shap
import pandas as pd
import numpy as np
def draw_shap(model,X_test_tensor,feature_names):

    # Set the global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False   
    feature_names = feature_names

    model.eval()
    # Use SHAP's DeepExplainer for feature importance assessment
    explainer = shap.GradientExplainer(model.cpu(), X_test_tensor)


    shap_values = explainer.shap_values(X_test_tensor)

    # Check the shape of shap_values: For multi-classification problems, shap_values might be a list
    # Select a certain category for visualization (assuming binary classification, select the first category)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  

    # Convert X_test_ex to NumPy format for use in SHAP plotting
    X_test_ex_np = X_test_tensor.detach().cpu().numpy()
    
    feature_names = feature_names.iloc[:, 1:(X_test_ex_np.shape[1]+1)].T
    print(X_test_ex_np.shape[1], len(feature_names))

    # Make sure that the number of columns of feature_names is consistent with that of X_test_ex
    assert len(feature_names) == X_test_ex_np.shape[1], "The feature names do not match the number of features in the data"

    feature_names = np.array(feature_names)

    # Draw the bar chart of the average feature importance of SHAP
    shap.summary_plot(shap_values[:, :, 0], X_test_ex_np, feature_names=feature_names)


    # Calculate the average of the absolute values of the SHAP values of each feature
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

    # Ensure that the mean_abs_shap_values are one-dimensional
    if mean_abs_shap_values.ndim > 1:
        mean_abs_shap_values = mean_abs_shap_values[:, 0]


    mean_abs_shap_df = pd.DataFrame(mean_abs_shap_values, index=feature_names, columns=['Mean |SHAP Value|'])

    # Sort the features according to the average SHAP value and select the top 20 important features
    mean_abs_shap_df = mean_abs_shap_df.sort_values(by='Mean |SHAP Value|', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    mean_abs_shap_df.plot(kind='barh', legend=False, color='skyblue')
    plt.title('Mean Absolute SHAP Values for Each Feature')
    plt.xlabel('Mean |SHAP Value|')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis() 
    plt.tight_layout()
    plt.show()
