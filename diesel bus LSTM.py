import pandas as pd
from pandas import concat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
import os
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
import numpy as np


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
plt.rcParams["font.sans-serif"] = ["SimHei"]   # 解决中文乱码问题
plt.rcParams["axes.unicode_minus"] = False    # 该语句解决图像中的“-”负号的乱码问题

df = pd.read_csv(r'D:\WORK\生态驾驶\项目\燃油车数据\2013年1月100路公交实验数据\1.5\1.5原始数据.csv', encoding='gbk')

# 相关性分析
data_corr = df.loc[:, ['Fuel Rate(gal/s)', 'acc(m/s2)', 'speed(m/s)', '坡度']]
data_corr.columns = ['Energy consumption', 'Acceleration', 'Speed', 'Road grade']
correlation = data_corr.corr(method='pearson')
plt.figure()
# 绘制热力图
sns.heatmap(correlation, linewidths=0.2, vmax=1, vmin=-1, linecolor='w', annot=True, annot_kws={'size': 10}, square=True)
plt.show()
plt.close()


# 将无监督转化为有监督
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]              # 变量个数
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(axis=0, inplace=True)
    return agg


tscv = TimeSeriesSplit(n_splits=3, test_size=0.3)
print(tscv)

data = df.loc[:, ['Fuel Rate(gal/s)', 'acc(m/s2)', 'speed(m/s)', '坡度']]
values = data.values
# 确保所有变量都是实数型
values = values.astype('float32')
# 将数据进行归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 将时间序列数据转换成有监督学习数据
reframed = series_to_supervised(scaled, 3, 1)
reframed.drop(columns=['var1(t-3)', 'var1(t-2)', 'var1(t-1)'], inplace=True)
# 将数据分割为训练集和测试集
values = reframed.values
n_train_num = int(len(values[:, 0]) * 0.8)       # 训练集数据个数
train = values[:n_train_num, :]
test = values[n_train_num:, :]
# 分离出特征集与标签
train_X, train_y = train[:, :-4], train[:, -4]
test_X, test_y = test[:, :-4], test[:, -4]
# 转换成3维数组 [样本数, 时间步长, 特征数]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, type(test_y))
print(train_X)


# # 创建模型
# model_y = Sequential()
# model_y.add(LSTM(128, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer='he_uniform'))
# model_y.add(Dropout(0.2))
# model_y.add(Dense(1, activation='tanh'))
# model_y.compile(loss='mae', optimizer='RMSProp')
# model_y.summary()

# # cross-validation
# values_X, values_y = values[:, :-4], values[:, -4]
# values_X = values_X.reshape((values_X.shape[0], 1, values_X.shape[1]))
# scores = cross_val_score(model_y, values_X, values_y, cv=tscv)
# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
#
# # 训练模型
# history_y = model_y.fit(train_X, train_y, epochs=200, batch_size=64, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# # 存储模型
# model_y.save('diesel bus LSTM.model')
# # 对损失进行可视化
# plt.plot(history_y.history['loss'], label='Train')
# plt.plot(history_y.history['val_loss'], label='Test')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Model Train Loss')
# plt.legend(loc='upper right')
# plt.show()

# 加载模型
model_y = load_model('diesel bus LSTM.model')

# 模型训练结果展示
time_begin = time.time()
prediction_y = model_y.predict(test_X)
prediction_y = prediction_y * (df.loc[:, "Fuel Rate(gal/s)"].max() - df.loc[:, "Fuel Rate(gal/s)"].min()) + df.loc[:, "Fuel Rate(gal/s)"].min()
time_end = time.time()
print('预测花费的时间为：', time_end-time_begin)
test_y = test_y * (df.loc[:, "Fuel Rate(gal/s)"].max() - df.loc[:, "Fuel Rate(gal/s)"].min()) + df.loc[:, "Fuel Rate(gal/s)"].min()


# 绘图
y1 = test_y
y2 = prediction_y
fig1, ax1 = plt.subplots(figsize=(14, 6))
x = range(0, len(prediction_y))
ax1.tick_params(labelsize=20)
plt.rcParams.update({'font.size': 15})     # 设置图例字体大小
df3 = pd.read_excel(r'D:\\WORK\\生态驾驶\\毕设\\学术成果\\TRB\\diesel bus power-based model.xlsx')
y3 = df3.loc[1:, 1]
df4 = pd.read_excel(r'D:\\WORK\\生态驾驶\\毕设\\学术成果\\TRB\\diesel bus XGBoost model.xlsx')
y4 = df4.loc[1:, 1]
plt.plot(x, y2, 'g', label="LSTM model")
plt.plot(x, y3, 'b', label="Power-based model", alpha=0.5)
plt.plot(x, y4, 'm', label="XGBoost model", alpha=0.5)
plt.plot(x, y1, 'r', label="Observed value", linestyle=':')
plt.xlabel("Time(s)", fontdict={'size': 20})
plt.ylabel("Energy Consumption(gallon)", fontdict={'size': 20})
plt.legend(loc='upper right')
# 显示网格
plt.grid(True)
# 限制横轴显示刻度的范围
plt.xlim(0, 1001)
plt.savefig("D:\\WORK\\生态驾驶\\毕设\\学术成果\\Part D\\图片\\diesel bus 3 models 带坡度.tiff", dpi=300, format="tiff")
plt.show()
plt.close()


# 组图
plt.figure(figsize=(14, 10))

ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2, colspan=1)
plt.rcParams.update({'font.size': 12})     # 设置图例字体大小
plt.plot(x, y2, 'g', label="LSTM model")
plt.plot(x, y3, 'r', label="Power-based model", alpha=0.5)
plt.plot(x, y4, 'b', label="XGBoost model", alpha=0.5)
plt.plot(x, y1, 'black', label="Observed value", linestyle=':')
plt.xlabel('Time(s)', size=20)
plt.ylabel('Energy consumption(gallon)', size=20)
# 显示网格
plt.grid(True)
# 设置图例的位置
plt.legend(loc='upper right')
# 限制横轴显示刻度的范围
plt.xlim(0, 1001)
# 图编号
plt.title('(a)', y=0.9, loc='left')

ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, colspan=1)
plt.rcParams.update({'font.size': 12})     # 设置图例字体大小
plt.scatter(y1, y2, c='g', label="LSTM model")
plt.scatter(y1, y3, c='r', label="Power-based model", alpha=0.5)
plt.scatter(y1, y4, c='b', label="XGBoost model", alpha=0.5)
plt.plot(y1, y1, c='black')
plt.xlabel('Observed value', size=20)
plt.ylabel('Predicted value', size=20)
# 显示网格
plt.grid(True)
# 设置图例的位置
plt.legend(loc='upper right')
# 图编号
plt.title('(b)',  y=0.85, loc='left')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

plt.savefig("D:\\WORK\\生态驾驶\\毕设\\学术成果\\Part D\\图片\\diesel bus 3 models two figures with grade.tiff", dpi=300, format="tiff")
plt.show()
plt.close()


# 预测误差
MSE = metrics.mean_squared_error(test_y, prediction_y)
RMSE = metrics.mean_squared_error(test_y, prediction_y)**0.5
MAE = metrics.mean_absolute_error(test_y, prediction_y)
r2 = r2_score(test_y, prediction_y, sample_weight=None, multioutput='uniform_average')
# MAPE = metrics.mean_absolute_percentage_error(test_y, prediction_y)    # 因为真实值有0，所以存在0除问题，该公式不可用
print(MSE, RMSE, MAE, r2)

prediction_y = model_y.predict(train_X)
r2 = r2_score(train_y, prediction_y, sample_weight=None, multioutput='uniform_average')
print('训练集r2:', r2)
