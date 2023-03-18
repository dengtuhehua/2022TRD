import pandas as pd
from pandas import concat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import joblib
import os
import seaborn as sns
from sklearn import metrics
import xgboost as xgb
from keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import r2_score
import shap


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
plt.rcParams["font.sans-serif"] = ["SimHei"]   # 解决中文乱码问题
plt.rcParams["axes.unicode_minus"] = False    # 该语句解决图像中的“-”负号的乱码问题

df_diesel = pd.read_csv(r'D:\WORK\生态驾驶\项目\燃油车数据\2013年1月100路公交实验数据\1.5\1.5原始数据.csv', encoding='gbk')
df_ebus = pd.read_excel(r'D:\WORK\生态驾驶\项目\LMEBEG1R8HE000054_20210428171832179_4239-52\处理后数据\带标签\下行完整_异常值处理后.xlsx')

# diesel bus
data_diesel = df_diesel.loc[:, ['Fuel Rate(gal/s)', 'acc(m/s2)', 'speed(m/s)', '坡度']]
train_data = data_diesel.loc[:, ['acc(m/s2)', 'speed(m/s)', '坡度']]
train_target = data_diesel.loc[:, 'Fuel Rate(gal/s)']
X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.2, random_state=1)    # 随机划分训练集

# 统一的测试集
n_train_num = int(len(data_diesel['Fuel Rate(gal/s)']) * 0.8)       # 训练集数据个数
test_X, test_y = train_data.loc[n_train_num:, ['acc(m/s2)', 'speed(m/s)', '坡度']], train_target[n_train_num:]

# 建立模型
# 网格搜索
# Rmse_squetion = []
# for i in range(1, 6):
#     for j in range(1, 6):
#         maxdepth = i * 10
#         lr = j*0.05
#         model = xgb.XGBRegressor(max_depth=maxdepth, learning_rate=lr, n_estimators=150, random_state=42)
#         model.fit(X_train, y_train)
#         test_pre = model.predict(X_test)
#         RMSE = metrics.mean_squared_error(y_test, test_pre) ** 0.5
#         Rmse_squetion.append(RMSE)
#
# outcome = pd.DataFrame(Rmse_squetion).values.reshape((5, 5))
# min_outcome = np.min(outcome)
# min_outcome_index = np.where(outcome == np.min(outcome))
# print(min_outcome, min_outcome_index)

# 最优模型
model = xgb.XGBRegressor(max_depth=20, learning_rate=0.10, n_estimators=200, random_state=42)
model.fit(X_train, y_train)
# 预测测试集
test_pre = model.predict(test_X)
# 保存预测结果
pd.DataFrame(test_pre).to_excel('D:\\WORK\\生态驾驶\\毕设\\学术成果\\TRB\\diesel bus XGBoost model.xlsx')

# 绘制预测值和真实值对比图
fig1, ax1 = plt.subplots(figsize=(14, 6))
x = range(0, len(test_pre))
ax1.tick_params(labelsize=20)
plt.rcParams.update({'font.size': 15})     # 设置图例字体大小
plt.plot(x, test_y, 'r', label="Observed value")
plt.plot(x, test_pre, 'g', label="Predicted value")
# plt.plot(x, y3, 'b', label="Two variables")
plt.xlabel("Time(s)", fontdict={'size': 20})
plt.ylabel("Fuel rate(gallon)", fontdict={'size': 20})
plt.legend(loc='upper right')
# 显示网格
plt.grid(True)
# 限制横轴显示刻度的范围
plt.xlim(0, 1001)
plt.savefig("D:\\WORK\\生态驾驶\\毕设\\学术成果\\Part D\\图片\\diesel bus XGBoost 带坡度.tiff", dpi=300, format="tiff")
plt.show()
plt.close()

# shap value
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)  # 传入特征矩阵X，计算SHAP值
# summarize the effects of all the features
shap.summary_plot(shap_values, X_train, plot_size=(6, 3))
shap.dependence_plot('acc(m/s2)', shap_values, X_train, interaction_index=None)
shap.dependence_plot('speed(m/s)', shap_values, X_train, interaction_index=None)


MSE = metrics.mean_squared_error(test_y, test_pre)
RMSE = metrics.mean_squared_error(test_y, test_pre)**0.5
MAE = metrics.mean_absolute_error(test_y, test_pre)
r2 = r2_score(test_y, test_pre, sample_weight=None, multioutput='uniform_average')
print('油车预测误差：', MSE, RMSE, MAE, r2)


# e-bus
data_ebus = df_ebus.loc[:, ['瞬时能耗/kwh', '车辆加速度/m/s2', '车辆速度/m/s', '坡度']]
train_data = df_ebus.loc[:, ['车辆加速度/m/s2', '车辆速度/m/s', '坡度']]
train_target = data_ebus.loc[:, '瞬时能耗/kwh']
X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.2, random_state=1)

# 统一的测试集
n_train_num = int(len(data_ebus['瞬时能耗/kwh']) * 0.8)       # 训练集数据个数
test_X, test_y = train_data.loc[n_train_num:, ['车辆加速度/m/s2', '车辆速度/m/s', '坡度']], train_target[n_train_num:]
print(type(test_X), type(test_y), test_X.shape)
# 建立模型
# 网格搜索
# Rmse_squetion = []
# for i in range(1, 6):
#     for j in range(1, 6):
#         maxdepth = i * 10
#         lr = j*0.05
#         model = xgb.XGBRegressor(max_depth=maxdepth, learning_rate=lr, n_estimators=250, random_state=42)
#         model.fit(X_train, y_train)
#         test_pre = model.predict(X_test)
#         RMSE = metrics.mean_squared_error(y_test, test_pre) ** 0.5
#         Rmse_squetion.append(RMSE)
#
# outcome = pd.DataFrame(Rmse_squetion).values.reshape((5, 5))
# min_outcome = np.min(outcome)
# min_outcome_index = np.where(outcome == np.min(outcome))
# print(min_outcome, min_outcome_index)

# 最优模型
model = xgb.XGBRegressor(max_depth=20, learning_rate=0.15, n_estimators=250, random_state=42)
model.fit(X_train, y_train)
# 保存模型
joblib.dump(model, "E-bus_XGBpre.joblib.dat")
# 预测测试集
test_pre = model.predict(test_X)
# 保存预测结果
pd.DataFrame(test_pre).to_excel('D:\\WORK\\生态驾驶\\毕设\\学术成果\\TRB\\e-bus XGBoost model.xlsx')

# 绘制预测值和真实值对比图
fig2, ax1 = plt.subplots(figsize=(14, 6))
x = range(0, len(test_pre))
ax1.tick_params(labelsize=20)
plt.rcParams.update({'font.size': 15})     # 设置图例字体大小
plt.plot(x, test_y, 'r', label="Observed value")
plt.plot(x, test_pre, 'g', label="Predicted value")
# plt.plot(x, y3, 'b', label="Two variables")
plt.xlabel("Time(s)", fontdict={'size': 20})
plt.ylabel("Energy consumption(kwh)", fontdict={'size': 20})
plt.legend(loc='upper right')
# 显示网格
plt.grid(True)
# 限制横轴显示刻度的范围
plt.xlim(0, 1001)
plt.savefig("D:\\WORK\\生态驾驶\\毕设\\学术成果\\Part D\\图片\\e-bus XGBoost 带坡度.tiff", dpi=300, format="tiff")
plt.show()
plt.close()


# shap value
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)  # 传入特征矩阵X，计算SHAP值
# summarize the effects of all the features
shap.summary_plot(shap_values, X_train, plot_size=(6, 3))
shap.dependence_plot('车辆加速度/m/s2', shap_values, X_train, interaction_index=None)
shap.dependence_plot('车辆速度/m/s', shap_values, X_train, interaction_index=None)

# 误差
MSE = metrics.mean_squared_error(test_y, test_pre)
RMSE = metrics.mean_squared_error(test_y, test_pre)**0.5
MAE = metrics.mean_absolute_error(test_y, test_pre)
r2 = r2_score(test_y, test_pre, sample_weight=None, multioutput='uniform_average')
print('电车预测误差：', MSE, RMSE, MAE, r2)

