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

df = pd.read_excel(r'D:\WORK\生态驾驶\项目\LMEBEG1R8HE000054_20210428171832179_4239-52\处理后数据\带标签\下行完整_异常值处理后.xlsx')

# SVM判断减速时是否存在动能回收
# 标签0表示无回收，1表示有回收
df['labelXGBoost'] = 0
dataXGBoost = df[df['车辆加速度/m/s2'] < 0]
dataXGBoost.loc[dataXGBoost['瞬时能耗/kwh'] < 0, 'labelXGBoost'] = 1
df.loc[df['瞬时能耗/kwh'] < 0, 'labelXGBoost'] = 1

# 归一化
dataXGBoost = dataXGBoost[['车辆加速度/m/s2', '车辆速度/m/s', '瞬时能耗/kwh', 'labelXGBoost']]
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler.fit(dataXGBoost)    # 找出每列最大、小值，并存储
datascaled = scaler.transform(dataXGBoost)
datascaled = pd.DataFrame(datascaled)
datascaled.columns = ('车辆加速度/m/s2', '车辆速度/m/s', '瞬时能耗/kwh', 'labelXGBoost')

# 将数据分割为训练集和测试集
values = datascaled.values
np.random.shuffle(values)
n_train = int(len(values[:, 0]) * 0.8)       # 训练集数据个数
train = values[:n_train, :]
test = values[n_train:, :]
train_data, test_data, train_label, test_label = train[:, 0:2], test[:, 0:2], train[:, 3], test[:, 3]
print(sum(test_label), type(test_data), test_data.shape, test_data, type(values))

# 生成分类器，网格搜索得到最佳参数C=5，gamma=30
# outcome_svm = []
# for i in range(25):
#     for j in range(6):
#         classifier = svm.SVC(C=0.2*(i+1), kernel='rbf', gamma=5*(j+1), decision_function_shape='ovo')    # ovr:一对多策略;ovo一对一
#         classifier.fit(train_data, train_label.ravel())    # ravel函数在降维时默认是行序优先
#         # 计算SVC分类器的准确率
#         print("训练集：", classifier.score(train_data, train_label))
#         print("测试集：", classifier.score(test_data, test_label))
#         # print(0.2*(i+1), 5*j, '\n')
#         outcome_svm.append(classifier.score(train_data, train_label)+classifier.score(test_data, test_label))
# outcomesvm = pd.DataFrame(outcome_svm).values.reshape(25, 1)
# max_outcomesvm = np.max(outcomesvm)
# max_outcomesvm_index = np.where(outcomesvm == np.max(outcomesvm))
# print(max_outcomesvm, max_outcomesvm_index)

# 最佳模型
Classifier = svm.SVC(C=1000, kernel='rbf', gamma=20, decision_function_shape='ovo')    # ovr:一对多策略;ovo一对一
Classifier.fit(train_data, train_label.ravel())    # ravel函数在降维时默认是行序优先
# 计算SVC分类器的准确率
y_pred = Classifier.predict(test_data)
print("SVM训练集：", Classifier.score(train_data, train_label))
print("SVM测试集：", Classifier.score(test_data, test_label))
print(confusion_matrix(test_label, y_pred))
print(classification_report(test_label, y_pred))


# XGBoost分类
# 建模
Classifier_XGBoost = XGBClassifier(learning_rate=0.1)
eval_set = [(test_data, test_label)]
Classifier_XGBoost.fit(train_data, train_label, early_stopping_rounds=10, eval_metric='logloss', eval_set=eval_set, verbose=True)
# XGBoost准确率
print("XGBoost训练集：", Classifier_XGBoost.score(train_data, train_label))
print("XGBoost测试集：", Classifier_XGBoost.score(test_data, test_label))
# 混淆矩阵
y_pred = Classifier_XGBoost.predict(test_data)
print(confusion_matrix(test_label, y_pred))
print(classification_report(test_label, y_pred))
cm_matrix = metrics.confusion_matrix(test_label, y_pred)
class_names = ['no regression', 'regression']    # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
print("Accuracy:", metrics.accuracy_score(test_label, y_pred))

xgboost.plot_importance(Classifier_XGBoost)
plt.show()
plt.close()


# Logistic回归分类
# penalty: 过拟合解决参数（正则化参数）,l1或者l2，L1 / L2    默认是L2的正则化,一般使用L2就可以#了，但是如果还是过拟合，就可以用L1，如果特征较多希望略去不重要特征，就有L1。
# solver: 参数优化方式
# 当penalty为l1的时候，参数只能是：liblinear(坐标轴下降法)；因为L1正则化的损失函数不是连续可导，liblinear只支持OvR，不支持MvM，而多元逻辑回归时一般采用MVM
# nlbfgs和cg都是关于目标函数的二阶泰勒展开
# 当penalty为l2的时候，参数可以是：liblinear，lbfgs(拟牛顿法)、newton-cg(牛顿法变种)，sag(minibatch)
# 维度<10000时，lbfgs法比较好，   维度>10000时， cg法比较好，显卡计算的时候，lbfgs和cg都比seg快
# 1.  liblinear：使用了坐标轴下降法来迭代优化损失函数。2.  lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。3. newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。 4.  sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合样本多的。
# multi_class: 分类方式参数；参数可选: ovr(默认)、multinomial；这两种方式在二元分类问题中，效果是一样的；在多元分类问题中，效果不一样
# ovr: one-vs-rest， 对于多元分类的问题，先将其看做二元分类，分类完成后，再迭代对其中一类继续进行二元分类
# multinomial: many-vs-many（MVM）,对于多元分类问题，如果模型有T类，我们每次在所有的T类样本里面选择两类样本出来，不妨记为T1类和T2类，把所有的输出为T1和T2的样本放在一起，把T1作为正例，T2作为负例，进行二元逻辑回归，得到模型参数。我们一共需要T(T-1)/2次分类
# class_weight: 特征权重参数
# TODO: Logistic回归是一种分类算法，不能应用于回归中(也即是说对于传入模型的y值来讲，不能是float类型，必须是int类型)

Classifier = LogisticRegressionCV(multi_class='ovr', fit_intercept=True, class_weight='balanced', Cs=np.logspace(-2, 2, 20), cv=2, penalty='l2', solver='newton-cg', tol=0.01)
Classifier.fit(train_data, train_label)
# 计算Logistic分类器的准确率
y_pred = Classifier.predict(test_data)
print("Logistic训练集：", Classifier.score(train_data, train_label))
print("Logistic测试集：", Classifier.score(test_data, test_label))
print(confusion_matrix(test_label, y_pred))
print(classification_report(test_label, y_pred))

# 决策树分类器
Classifier = DecisionTreeClassifier()
Classifier.fit(train_data, train_label)
y_pred = Classifier.predict(test_data)
# 计算决策树分类器的准确率
print("决策树训练集：", Classifier.score(train_data, train_label))
print("决策树测试集：", Classifier.score(test_data, test_label))
print(confusion_matrix(test_label, y_pred))
print(classification_report(test_label, y_pred))

# 朴素贝叶斯分类器
# 创建高斯朴素贝叶斯实例
Classifier = GaussianNB()
# 使用sigmoid校准创建校准交叉验证
# Classifier_sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid')
# 校准的概率
Classifier.fit(train_data, train_label)
# 计算朴素贝叶斯分类器的准确率
y_pred = Classifier.predict(test_data)
print("朴素贝叶斯训练集：", Classifier.score(train_data, train_label))
print("朴素贝叶斯测试集：", Classifier.score(test_data, test_label))
print(confusion_matrix(test_label, y_pred))
print(classification_report(test_label, y_pred))


# 多元线性回归
df['x1'] = df['车辆速度/m/s']*df['sin']
df['x2'] = df['车辆速度/m/s']*df['cos']
df['x3'] = (df['车辆速度/m/s']**2)*df['cos']
df['x4'] = df['车辆速度/m/s']**3
df['x5'] = df['车辆速度/m/s']*df['车辆加速度/m/s2']
df['y'] = df['瞬时能耗/kwh']
# 无动能回收0
df_nonregress = df[df['labelXGBoost'] == 0]
n_train_num = int(len(df_nonregress['瞬时能耗/kwh']) * 0.8)     # 训练集数据个数
df_nonregress_train = df_nonregress.loc[:n_train_num, ['x1', 'x2', 'x3', 'x4', 'x5', 'y']]
df_nonregress_test = df_nonregress.loc[n_train_num:, ['x1', 'x2', 'x3', 'x4', 'x5', 'y']]
model1 = LinearRegression()
model1.fit(df_nonregress_train.loc[:, ['x1', 'x2', 'x3', 'x4', 'x5']], df_nonregress_train.loc[:, ['y']])
a = model1.intercept_  # 截距
b = model1.coef_  # 回归系数
print("最佳拟合线:截距", a, ",回归系数：", b)
score = model1.score(df_nonregress_train.loc[:, ['x1', 'x2', 'x3', 'x4', 'x5']], df_nonregress_train.loc[:, ['y']])
print('R\u00b2:', score)
# 对测试集进行预测
Y_pred = model1.predict(df_nonregress_test.loc[:, ['x1', 'x2', 'x3', 'x4', 'x5']])
print(Y_pred)
plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
plt.plot(range(len(Y_pred)), df_nonregress_test.loc[:, ['y']], 'r', label="true")
plt.legend()
plt.xlim(0, 1000)
plt.show()
plt.close()
# 输出r2
r2 = r2_score(df_nonregress_test.loc[:, ['y']], Y_pred, sample_weight=None, multioutput='uniform_average')
print('无回收状态测试集r2：', r2)

# 有动能回收1
df_regress = df[df['labelXGBoost'] == 1]
n_train_num = int(len(df_regress['瞬时能耗/kwh']) * 0.8)     # 训练集数据个数
df_regress_train = df_regress.loc[:n_train_num, ['x1', 'x2', 'x3', 'x4', 'x5', 'y']]
df_regress_test = df_regress.loc[n_train_num:, ['x1', 'x2', 'x3', 'x4', 'x5', 'y']]
model2 = LinearRegression()
model2.fit(df_regress_train.loc[:, ['x1', 'x2', 'x3', 'x4', 'x5']], df_regress_train.loc[:, ['y']])
a = model2.intercept_  # 截距
b = model2.coef_  # 回归系数
print("最佳拟合线:截距", a, ",回归系数：", b)
score = model2.score(df_regress_train.loc[:, ['x1', 'x2', 'x3', 'x4', 'x5']], df_regress_train.loc[:, ['y']])
print('R\u00b2:', score)
# 对测试集进行预测
Y_pred = model2.predict(df_regress_test.loc[:, ['x1', 'x2', 'x3', 'x4', 'x5']])
print(Y_pred)
plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
plt.plot(range(len(Y_pred)), df_regress_test.loc[:, ['y']], 'r', label="true")
plt.legend()
plt.xlim(0, 1000)
plt.show()
plt.close()
# 输出r2
r2 = r2_score(df_regress_test.loc[:, ['y']], Y_pred, sample_weight=None, multioutput='uniform_average')
print('有回收状态测试集r2：', r2)

# 预测测试集
n_train_num = int(0.8*len(df['labelXGBoost']))       # 训练集数据个数
df_test = df.loc[n_train_num:, ['车辆加速度/m/s2', '车辆速度/m/s', 'x1', 'x2', 'x3', 'x4', 'x5', 'y', 'labelXGBoost']]
df_test = df_test.reset_index(drop=True)
print(df_test)
Y_prediction = []
y_noregre = model1.predict(np.array(df_test.loc[:, ['x1', 'x2', 'x3', 'x4', 'x5']]).reshape(-1, 5))
y_regre = model2.predict(np.array(df_test.loc[:, ['x1', 'x2', 'x3', 'x4', 'x5']]).reshape(-1, 5))

min_acc = min(dataXGBoost.loc[:, '车辆加速度/m/s2'])
max_acc = max(dataXGBoost.loc[:, '车辆加速度/m/s2'])
min_v = min(dataXGBoost.loc[:, '车辆速度/m/s'])
max_v = max(dataXGBoost.loc[:, '车辆速度/m/s'])
time_begin = time.time()
for i in range(len(df_test['x1'])):
    acc_ = (df_test.loc[i, '车辆加速度/m/s2'] - min_acc)/(max_acc - min_acc)
    v_ = (df_test.loc[i, '车辆速度/m/s'] - min_v)/(max_v - min_v)
    x = np.append(acc_, v_).reshape(1, -1)
    labelxg = Classifier_XGBoost.predict(x)
    # print(labelxg)
    if labelxg == 0:
        # y = model1.predict(np.array(df.loc[i, ['x1', 'x2', 'x3', 'x4']]).reshape(1, 4))[0][0]
        y = y_noregre[i]
    else:
        # y = model1.predict(np.array(df.loc[i, ['x1', 'x2', 'x3', 'x4']]).reshape(1, 4))[0][0]
        y = y_regre[i]
    Y_prediction.append(y)
time_end = time.time()
print('预测花费的时间为：', time_end-time_begin)

# 输出预测结果
print(Y_prediction)
print(df_test['y'])
# 保存预测结果
# pd.DataFrame(Y_prediction).to_excel('D:\\WORK\\生态驾驶\\毕设\\学术成果\\TRB\\e-bus power-based model.xlsx')

# 绘制电车瞬时能耗和预测能耗对比图
y1 = df_test['y']
y2 = Y_prediction
df3 = pd.read_excel(r'D:\\WORK\\生态驾驶\\毕设\\学术成果\\TRB\\noclassification.xlsx')
y3 = df3.loc[:, 1]
len_x = range(len(y1))
fig1, ax1 = plt.subplots(figsize=(14, 6))
ax1.tick_params(labelsize=20)
plt.rcParams.update({'font.size': 15})     # 设置图例字体大小
plt.plot(len_x, y2, 'g', label="With classification")
plt.plot(len_x, y3, 'b', label="Without classification")
plt.plot(len_x, y1, 'r', label="Observed value", linestyle=':')
plt.xlabel('Time(s)', size=20)
plt.ylabel('Energy consumption(kwh)', size=20)
# 显示网格
plt.grid(True)
# 设置图例的位置
plt.legend(loc='upper right')
# 限制横轴显示刻度的范围
plt.xlim(0, 1001)
# plt.savefig("D:\\WORK\\生态驾驶\\毕设\\学术成果\\Part D\\图片\\e-bus two power-based model linear.tiff", dpi=300, format="tiff")
plt.show()
plt.close()


# 组图
plt.figure(figsize=(14, 10))

ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2, colspan=1)
plt.rcParams.update({'font.size': 12})     # 设置图例字体大小
# plt.plot(len_x, y2, 'r', label="With XGBoost-based classifier")
# plt.plot(len_x, y3, 'b', label="Without XGBoost-based classifier")
# plt.plot(len_x, y1, 'black', label="Observed value", linestyle=':')
plt.plot(len_x, y2, 'r', label="有XGBoost分类器")
plt.plot(len_x, y3, 'b', label="无XGBoost分类器")
plt.plot(len_x, y1, 'black', label="真实值", linestyle=':')
plt.xlabel('时间(s)', size=20)
plt.ylabel('能耗(kwh)', size=20)
# 显示网格
plt.grid(True)
# 设置图例的位置
plt.legend(loc='lower right')
# 限制横轴显示刻度的范围
plt.xlim(0, 1001)
# 图编号
plt.title('(a)', y=0.9, loc='left')

ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, colspan=1)
plt.rcParams.update({'font.size': 12})     # 设置图例字体大小
plt.scatter(y1, y2, c='r', label="有XGBoost分类器")
plt.scatter(y1, y3, c='b', label="无XGBoost分类器")
plt.plot(y1, y1, c='black')
plt.xlabel('真实值', size=20)
plt.ylabel('预测值', size=20)
# 显示网格
plt.grid(True)
# 设置图例的位置
plt.legend(loc='lower right')
# 图编号
plt.title('(b)',  y=0.85, loc='left')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

# plt.savefig("D:\\WORK\\生态驾驶\\毕设\\硕士论文\\图片\\电动公交车功率模型组图.jpg", dpi=300, format="jpg")
plt.show()
plt.close()


# 预测误差输出
MSE = metrics.mean_squared_error(y1, y2)
RMSE = metrics.mean_squared_error(y1, y2)**0.5
MAE = metrics.mean_absolute_error(y1, y2)
r2 = r2_score(y1, y2, sample_weight=None, multioutput='uniform_average')
print(MSE, RMSE, MAE, r2)
