import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
import matplotlib
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=None, shuffle=True) # 5Folds
matplotlib.use('TkAgg')
from sklearn.linear_model import Lasso

from lassonet import BaseLassoNet
import torch
from torch.nn import functional as F
import warnings
from scipy.stats import norm
# Import matplotlib for plotting graphs ans seaborn for attractive graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from matplotlib.lines import Line2D
import shap
from sklearn.manifold import TSNE
def plot(x, colors):
    # Set the font
    plt.rcParams["font.family"] = "Times New Roman"
    padding = 0.1
    target_ratio = 4 / 5
    # Define the color palette and category names
    palette = np.array(sns.color_palette("pastel", 2))
    labels = ["HER2-Unchanged", "HER2-changed"]
    # Create graphics
    fig, ax = plt.subplots(figsize=(10, 8))
    # Draw the data points of each category and add legend labels
    for i, label in enumerate(labels):
        ax.scatter(
            x[colors == i, 0], x[colors == i, 1],
            lw=0, s=40, color=palette[i], label=label
        )


    # Calculate the data range
    x_min, x_max = x[:, 0].min(), x[:, 0].max()
    y_min, y_max = x[:, 1].min(), x[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    # Add margins
    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding
    # Adjust the range according to the target proportion
    current_ratio = (y_max - y_min) / (x_max - x_min)
    if current_ratio < target_ratio:
        new_y_range = (x_max - x_min) * target_ratio
        y_center = (y_max + y_min) / 2
        y_min, y_max = y_center - new_y_range / 2, y_center + new_y_range / 2
    else:
        new_x_range = (y_max - y_min) / target_ratio
        x_center = (x_max + x_min) / 2
        x_min, x_max = x_center - new_x_range / 2, x_center + new_x_range / 2

    # Set the adjusted coordinate axis range and aspect ratio
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect(target_ratio)  

    # Automatically calculate the appropriate coordinate axis intervals
    x_ticks = np.linspace(x_min, x_max, num=6)
    y_ticks = np.linspace(y_min, y_max, num=6)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Add legends and represent the categories with large dot blocks
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i],
               markersize=15, label=labels[i]) for i in range(2)
    ]
    ax.legend(handles=legend_elements, title="HER2 STATUS", loc="upper right", fontsize=11)

    ax.text(0.95, 0.05, "HKFSNet", transform=ax.transAxes, fontsize=16,
            weight='bold', color='black', ha='right', va='bottom',
            path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])
    plt.show()
    plt.savefig('./output_with_aspect_and_legend.png', dpi=300, bbox_inches='tight') 
    return fig, ax
def soft_threshold(l, x):
    return torch.sign(x) * torch.relu(torch.abs(x) - l)
def sign_binary(x):
    ones = torch.ones_like(x)
    return torch.where(x >= 0, ones, -ones)
warnings.filterwarnings('ignore')
tprs=[]
aucs=[]
mean_fpr = np.linspace(0, 1, 100)
under_sampler = RandomUnderSampler(random_state=2)

num_epochs = 140        #520  500 600  470   316 480 390 379 375  240  130 100
split_random_state1 = 4    #test
split_random_state2 = 21   #val
seed = 4016             #53 47      194   5  3407  372
gate = 0.4
hidden_size = 220


def HKFS(train_data_original, alpha, label, M = 10):

    x, y = train_data_original.iloc[:, 1:], label
    alpha = alpha
    HKFS = Lasso(alpha=alpha, random_state=1, max_iter=M)#10000
    HKFS.fit(x, y)
    mask = HKFS.coef_ != 0.0
    # Select the features
    x = x.loc[:, mask]
    # result = pd.concat([x, y], axis=1)
    train_data = pd.concat([train_data_original['namelist'], x], axis=1)
    return train_data

'''
train: val: test = 7 : 1 : 2 
'''
train_data_value_A = pd.read_excel(r'.\datasets\BUS_UCLM.xlsx', header=0, usecols="A:B", sheet_name="namelist")#usecols="B:AGF"
train_label_A = pd.concat([pd.DataFrame(np.zeros(264))], axis=0)
X_train_val_name, X_test_name, _, _ = train_test_split(train_data_value_A, train_label_A, test_size=0.2, random_state=split_random_state1)#7 :72
X_train_val_label_A = pd.concat([pd.DataFrame(np.zeros(211))], axis=0)
X_train_name, X_val_name, _, _ = train_test_split(X_train_val_name, X_train_val_label_A, test_size=0.125, random_state=split_random_state2)#7 :72
train_data_original = pd.read_excel(r'.\datasets\BUS_UCLM.xlsx', header=0, sheet_name="All")#usecols="B:AGF"
# train_data = train_data_original
labels = pd.read_excel(r'.\datasets\BUS_UCLM.xlsx', header=0, usecols="B", sheet_name="namelist")
train_data = HKFS(train_data_original, 0.09117, labels)
print(train_data.shape)

extracted_train_data = train_data[train_data['namelist'].isin(X_train_name['namelist'])]
extracted_val_data = train_data[train_data['namelist'].isin(X_val_name['namelist'])]
extracted_test_data = train_data[train_data['namelist'].isin(X_test_name['namelist'])]

extracted_train_label = extracted_train_data[extracted_train_data['namelist'].isin(X_train_name['namelist'])]
extracted_val_label = extracted_val_data[extracted_val_data['namelist'].isin(X_val_name['namelist'])]
extracted_test_label = extracted_test_data[extracted_test_data['namelist'].isin(X_test_name['namelist'])]
# Create a dictionary and map the labels to the corresponding values
label_mapping = dict(zip(train_data_value_A['namelist'], train_data_value_A['labels']))
# Use the map function to map the contents in the data table to the values of the labels
extracted_train_label['labels'] = extracted_train_label['namelist'].map(label_mapping)
extracted_val_label['labels'] = extracted_val_label['namelist'].map(label_mapping)
extracted_test_label['labels'] = extracted_test_label['namelist'].map(label_mapping)


X_train = extracted_train_data.iloc[:, 1:]
X_val = extracted_val_data.iloc[:, 1:]
X_test = extracted_test_data.iloc[:, 1:]

y_train = extracted_train_label['labels']
y_val = extracted_val_label['labels']
y_test = extracted_test_label['labels']

# Standardized data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
X_test = scaler.fit_transform(X_test)

# Perform exclusive heat coding on the labels
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Convert to PyTorch Tensor
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Train the model
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_Size = [550, 300, 300]  # 550    2 150
# batch_Size = [220, 100]
# batch_Size = [220, 120]  #100-110  80+

train_loader = DataLoader(train_dataset, batch_size=batch_Size[0], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_Size[1], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_Size[2], shuffle=True)


# Define the URAN model
class URAN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size ,channel=2,reduction=16):
        super(URAN, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu2 = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)  
        self.dropout = nn.Dropout(p=0.4)
    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc3(x)
        x = self.relu2(x)

        input1 = x.unsqueeze(-1).unsqueeze(-1)
        b, c, _, _ = input1.size()
        y1 = self.avg_pool(input1).view(b, c)
        y2 = self.max_pool(input1).view(b, c)
        y = (y1+y2).view(b, c)
        y = self.fc(y).view(b, c,1,1)
        x = (input1 * y.expand_as(input1)).squeeze()

        x = self.softmax(x)
        x = self.dropout(x)
        return x


def Sensitivity_score(Y_test, Y_pred, n):  

    sen = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)

    return sen


def Specificity_score(Y_test, Y_pred, n):
    spe = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)

    return spe

# Model initialization
input_size = X_train.shape[1]  
# hidden_size = 200         #185   220  262  219
output_size = 2  

i = 1
accuracy_list = list()
recall_list = list()
precision_list = list()
f1_list = list()
auc_score_list = list()
best_metric = 0


# for seed in range(100, 2000):
for b in range(4, 5):
    print(f"{b+1}_fold")
    seed = b

    # Set the seeds in the CPU and generate random numbers
    # ======Set random seeds==============
    '''
    1：0.61 42：0.57 0: 59 2:59.13 3:61.2912% 4: 60.11% 5:58.91% 6: 63.17% 7:54.91% 8:56.94% 9: 51.65% 10:57.72%
    11:58 12:60 13:56 14:59.12% 15:56.78% 16:60.46% 17:52.66% 18:52.86%  19:56.54% 
    '''

    torch.manual_seed(seed)
    # Divide the training set and the test set for cross-validation five times
    for split_num in range(5):
        print('split_num=', split_num)
        model = URAN(input_size, hidden_size, output_size)
        model.__init__(input_size, hidden_size, output_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00005)#lr=0.0006)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0  
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                # optimizer1.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y.float()) 
                loss.backward()
                optimizer.step()
                # optimizer1.step()
                total_loss += loss.item()

            if epoch % 2 == 0:
                average_loss = total_loss / len(train_loader)
                # print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss:.4f}')
                # Calculate and print the accuracy of the training set
                model.eval()
                with torch.no_grad():
                    correct_train = 0
                    total_train = 0
                    for batch_X, batch_y in train_loader:
                        output_train = model(batch_X)
                        _, predicted_train = torch.max(output_train.data, 1)
                        total_train += batch_y.size(0)
                        batch_y = np.argmax(batch_y, axis=1)
                        correct_train += (predicted_train == batch_y).sum().item()


                    accuracy_train = correct_train / total_train
                    y_pred = np.argmax(output_train.data, axis=1)

                    auc_score = roc_auc_score(batch_y, y_pred, multi_class='ovr')
                    print(f'Training Accuracy: {accuracy_train * 100:.2f}%,Training AUC: {auc_score :.4f}')
            # if epoch % 10 == 0:
            #     for batch_X, batch_y in train_loader:
            #         outputs = model(batch_X)
            #         batch_y = np.argmax(batch_y, axis=1)
            #         X_tSNE = outputs.detach().cpu().numpy()
            #         y_tSNE = batch_y.detach().cpu().numpy()
            #         # Implementing the TSNE Function - ah Scikit learn makes it so easy!
            #         digits_final = TSNE(perplexity=30).fit_transform(X_tSNE)
            #         # Play around with varying the parameters like perplexity, random_state to get different plots
            #         plot(digits_final, y_tSNE)


        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_X, batch_y in val_loader:
                output = model(batch_X)
                _, predicted = torch.max(output.data, 1)

                batch_y = np.argmax(batch_y, axis=1)
                # print(f'Predicted Labels: {predicted.tolist()}')
                # print(f'Actual Labels   : {batch_y.tolist()}')

                accuracy = accuracy_score(batch_y, predicted)
                recall = recall_score(batch_y, predicted, average='macro')
                precision = precision_score(batch_y, predicted, average='macro')
                f1 = f1_score(batch_y, predicted, average='macro')

                y_pred = np.argmax(output.data, axis=1)
                auc_score = roc_auc_score(batch_y, y_pred, multi_class='ovr')

            i += 1


        # Model saving
        metric = auc_score
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            save_dir = 'checkpoints/checkpoint_0320/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = save_dir + str(epoch + 1) + f"best_AUC_{auc_score}_model.pth"
            if auc_score > 0.8:
                torch.save(model.state_dict(), save_path)

                print('saved new best metric model')


        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_X, batch_y in test_loader:
                output = model(batch_X)
                _, predicted = torch.max(output.data, 1)

                batch_y = np.argmax(batch_y, axis=1)
                # print(f'Predicted Labels: {predicted.tolist()}')
                # print(f'Actual Labels   : {batch_y.tolist()}')

            accuracy = accuracy_score(batch_y, predicted)
            recall = recall_score(batch_y, predicted, average='micro')
            precision = precision_score(batch_y, predicted, average='macro')
            f1 = f1_score(batch_y, predicted, average='macro')

            Specificity = Specificity_score(batch_y, predicted, 2)

            y_pred = np.argmax(output.data, axis=1)
            auc_score = roc_auc_score(batch_y, y_pred, multi_class='ovr')

            # print(f'Predicted Labels in test: {predicted.tolist()}')
            # print(f'Actual Labels in test   : {batch_y.tolist()}')
            # print(f'num_epochs: {num_epochs}')
            # print(f'batch_Size: {batch_Size[2]}')
            print(f"test Accuracy: {accuracy* 100:.4f}%")
            print(f"test Sensitivity: {recall* 100:.4f}%")
            print(f"test Precision: {precision* 100:.4f}%")
            print(f"test F1-Score: {f1* 100:.4f}%\n")
            print("test AUC:", auc_score)

# print(f"val_AUC: {np.mean(auc_score_array)* 100:.2f}% internal_test_AUC:{auc_score* 100:.2f}%")
# accuracy = correct / total
# print(f'Test Accuracy: {accuracy * 100:.2f}%')
            accuracy_list.append(accuracy)
            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)
            auc_score_list.append(auc_score)

    auc_interval = norm.interval(0.95, loc=np.mean(auc_score_list), scale=np.std(auc_score_list) / np.sqrt(len(auc_score_list)))
    accuracy_interval = norm.interval(0.95, loc=np.mean(accuracy_list),
                                      scale=np.std(accuracy_list) / np.sqrt(len(accuracy_list)))
    recalls_interval = norm.interval(0.95, loc=np.mean(recall_list), scale=np.std(recall_list) / np.sqrt(len(recall_list)))
    F1_scores_interval = norm.interval(0.95, loc=np.mean(f1_list),
                                       scale=np.std(f1_list) / np.sqrt(len(f1_list)))
    Precisions_interval = norm.interval(0.95, loc=np.mean(precision_list),
                                        scale=np.std(precision_list) / np.sqrt(len(precision_list)))
    # print('learning_rate =', learning_rate)
    print(f"Average auc: {np.mean(auc_score_list):.4f}±{np.std(auc_score_list) / np.sqrt(len(auc_score_list)):.4f} 95% CI: {auc_interval}")
    print(
        f"Average accuracy: {np.mean(accuracy_list):.4f}±{np.std(accuracy_list) / np.sqrt(len(accuracy_list)):.4f} 95% CI: {accuracy_interval}")
    print(
        f"Average recalls: {np.mean(recall_list):.4f}±{np.std(recall_list) / np.sqrt(len(recall_list)):.4f} 95% CI: {recalls_interval}")
    print(
        f"Average F1_score: {np.mean(f1_list):.4f}±{np.std(f1_list) / np.sqrt(len(f1_list)):.4f}  95% CI: {F1_scores_interval}")
    print(
        f"Average Precisions: {np.mean(precision_list):.4f}±{np.std(precision_list) / np.sqrt(len(precision_list)):.4f}  95% CI: {Precisions_interval}")


# # 假设你有一个特征名称列表
# feature_names = pd.read_excel(r'.\417feature\label1\label1.xlsx', header=None, nrows=1)#usecols="B:AGF"
# from draw_shap import draw_shap
# draw_shap(model=model, feature_names=feature_names, X_test_tensor=X_test_tensor)
