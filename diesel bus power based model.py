import numpy as np
import pandas as pd
from pylab import *
from sklearn import svm
import sklearn
import matplotlib.ticker as mticker
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.linear_model import LinearRegression
import xgboost
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.metrics import r2_score
import random
import shap


plt.rcParams["font.sans-serif"] = ["SimHei"]   # 解决中文乱码问题
plt.rcParams["axes.unicode_minus"] = False    # 该语句解决图像中的“-”负号的乱码问题

df = pd.read_csv(r'D:\WORK\生态驾驶\项目\燃油车数据\2013年1月100路公交实验数据\1.5\1.5原始数据.csv', encoding='gbk')
print(df)


# 多元线性回归
df['x1'] = df['speed(m/s)']*df['sin']
df['x2'] = df['speed(m/s)']*df['cos']
df['x3'] = (df['speed(m/s)']**2)*df['cos']
df['x4'] = df['speed(m/s)']**3
df['x5'] = df['speed(m/s)']*df['acc(m/s2)']
df['y'] = df['Fuel Rate(gal/s)']

n_train_num = int(len(df['Fuel Rate(gal/s)']) * 0.8)     # 训练集数据个数
df_train = df.loc[:n_train_num, ['x1', 'x2', 'x3', 'x4', 'x5', 'y']]
df_test = df.loc[n_train_num:, ['x1', 'x2', 'x3', 'x4', 'x5', 'y']]
model = LinearRegression()
model.fit(df_train.loc[:, ['x1', 'x2', 'x3', 'x4', 'x5']], df_train.loc[:, ['y']])
a = model.intercept_  # 截距
b = model.coef_  # 回归系数
print("最佳拟合线:截距", a, ",回归系数：", b)
score = model.score(df_train.loc[:, ['x1', 'x2', 'x3', 'x4', 'x5']], df_train.loc[:, ['y']])
print('R\u00b2:', score)
# 对测试集进行预测
Y_pred = model.predict(df_test.loc[:, ['x1', 'x2', 'x3', 'x4', 'x5']])
# 保存预测结果
# pd.DataFrame(Y_pred).to_excel('D:\\WORK\\生态驾驶\\毕设\\学术成果\\TRB\\diesel bus power-based model.xlsx')

# 绘图
fig1, ax1 = plt.subplots(figsize=(14, 6))
x = range(0, len(Y_pred))
ax1.tick_params(labelsize=20)
plt.rcParams.update({'font.size': 15})     # 设置图例字体大小
plt.plot(x, df_test.loc[:, ['y']], 'r', label="Observed value")
plt.plot(x, Y_pred, 'g', label="Predicted value")
plt.xlabel("Time(s)", fontdict={'size': 20})
plt.ylabel("Fuel rate(gallon)", fontdict={'size': 20})
plt.legend(loc='upper right')
plt.grid(True)
plt.xlim(0, 1001)
plt.savefig("D:\\WORK\\生态驾驶\\毕设\\学术成果\\Part D\\图片\\diesel bus power-based.tiff", dpi=300, format="tiff")
plt.show()
plt.close()


y1 = Y_pred
y2 = df_test.loc[:, ['y']]
# 预测误差输出
MSE = metrics.mean_squared_error(y1, y2)
RMSE = metrics.mean_squared_error(y1, y2)**0.5
MAE = metrics.mean_absolute_error(y1, y2)
r2 = r2_score(y1, y2, sample_weight=None, multioutput='uniform_average')
print(MSE, RMSE, MAE, r2)
