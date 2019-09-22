# 单颗决策数
# 导入需要的库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
# 导入数据
# pd.set_option('display.max_rows', 100, 'display.max_columns', 50)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data = pd.read_csv('tatandata.csv')
# print(data.info())
# print(data['Age'])
# print(data.isnull().sum())
data['Age'] = data['Age'].fillna(data['Age'].mean())
# print(data.isnull().sum())
# print(data['Sex'])
data['Sex'] = (data['Sex'] == 'male').astype('int')
# print(data['Sex'])
#  删除空值
data.drop(['Cabin', 'Ticket', 'Name'], inplace=True, axis=1)
data = data.dropna()

利用 map函数 处理分类值
# size_mapping = {'S': 1,
#                 'C': 2,
#                 'Q': 3}
# data['Embarked'] = data['Embarked'].map(size_mapping)
# 三分类转换数值变量,将数组转换成列表.tolist()
labels = data['Embarked'].unique().tolist()

data['Embarked'] = data['Embarked'].apply(lambda x: labels.index(x))
# print(data['Embarked'])
# 提取 特征和标签
x = data.iloc[:, data.columns != 'Survived']
y = data.iloc[:, data.columns == 'Survived']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# 修正索引
for i in [x_train, x_test, y_train, y_test]:
    i.index = range(i.shape[0])

# print(x_train.head())

# 建立模型
clf = DecisionTreeClassifier(random_state=25)
clf = clf.fit(x_train, y_train)

# 预测
score_ = clf.score(x_test, y_test)
# print(score_)
score = cross_val_score(clf, x, y, cv=10).mean()
# print(score)
# 0.7790262172284644
# 0.7469611848825333

# 调整参数
# tr, te = [], []
# for i in range(10):
#     clf = DecisionTreeClassifier(criterion='gini', max_depth=i+1, random_state=25)
#     clf.fit(x_train, y_train)
#     score_tr = clf.score(x_train, y_train)
#     score_te = cross_val_score(clf, x, y, cv=10).mean()
#     tr.append(score_tr)
#     te.append(score_te)
# print(max(tr), max(te))
# plt.plot(range(1, 11), tr, c='r', label='train')
# plt.plot(range(1, 11), te, c='g', label='test')
# plt.legend()
# plt.show()

# tr, te = [], []
# for i in range(10):
#     clf = DecisionTreeClassifier(criterion='entropy', max_depth=i+1, random_state=25)
#     clf.fit(x_train, y_train)
#     score_tr = clf.score(x_train, y_train)
#     score_te = cross_val_score(clf, x, y, cv=10).mean()
#     tr.append(score_tr)
#     te.append(score_te)
# print(max(tr), max(te))
# plt.plot(range(1, 11), tr, c='r', label='train')
# plt.plot(range(1, 11), te, c='g', label='test')
# plt.legend()
# plt.show()
# 0.9405144694533762 0.8143896833503576
# 0.909967845659164 0.8166624106230849

# 网格搜索
# parameters = {'splitter': ('best', 'random'),
#               'criterion': ('gini', 'entropy'),
#               'max_depth': ([3, 4, 5, 6, 7, 8, 9, 10]),
#               'min_samples_leaf': ([1, 2, 3, 4, 5, 6])}
# clf = DecisionTreeClassifier()
# GS = GridSearchCV(clf, parameters, cv=10)
# GS.fit(x_train, y_train)
# print(GS.best_params_)
# print(GS.best_score_)
# {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'splitter': 'best'}
# 0.8215434083601286

# 随机森林
from sklearn.datasets import load_boston
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
# dataset = load_boston()
# # print(dataset.data.shape)
# 特征和标签
# x_full, y_full = dataset.data, dataset.target
# n_samples = x_full.shape[0]
# n_features = x_full.shape[1]
# # print(n_samples, n_features)
#
# rng = np.random.RandomState(0)
# missing_rate = 0.5
# n_missing_samples = int(np.floor(n_samples*n_features*missing_rate))
#
# # print(n_missing_samples)
# missing_features = rng.randint(0, n_features, n_missing_samples)
# missing_samples = rng.randint(0, n_samples, n_missing_samples)
# # print(missing_features)
#
# x_missing = x_full.copy()
# y_missing = y_full.copy()
#
# # print(x_missing.shape)
# # print(y_missing.shape)
#
#
# x_missing[missing_samples, missing_features] = np.nan
# # print(x_missing.shape)
# x_missing = pd.DataFrame(x_missing)
# # print(x_missing.shape)
# # print(x_missing.head(2))

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
data = load_breast_cancer()
x = data.data
y = data.target


# n_estimators 调参
# per_s = []
# for i in range(35, 49):
#     rfc = RandomForestClassifier(n_estimators=i, random_state=90)
#     pre_score = cross_val_score(rfc, x, y, cv=10).mean()
#     per_s.append(pre_score)
#     print('这是%d棵树' % i, pre_score)
# plt.plot(range(10, 71), per_s, c='r')
# plt.show()

# 这是39棵树 0.9719568317345088
# 这是45棵树 0.9719870797683866


# 调整max_depth
# param_grid = {'max_depth': np.arange(1, 20, 1)}
# rfc = RandomForestClassifier(n_estimators=45, random_state=90)
# gri = GridSearchCV(rfc, param_grid, cv=10)
# gri.fit(x, y)
# print(gri.best_params_)
# print(gri.best_score_)
# {'max_depth': 11}
# 0.9718804920913884

# param_grid = {'max_features': np.arange(5, 30, 1)}
# rfc = RandomForestClassifier(n_estimators=45,  random_state=90)
# gri = GridSearchCV(rfc, param_grid, cv=10)
# gri.fit(x, y)
# print(gri.best_params_)
# print(gri.best_score_)
# {'max_features': 5}
# 0.9718804920913884
# maxfeatrue的增加会导致误差变大，前面maxdepth的变大也会增加误差，说明模型已经在上限了，没有参数可以左右了。
# param_grid = {'min_sample_leaf': np.arange(1, 1+10, 1)}
# rfc = RandomForestClassifier(n_estimators=45, random_state=90)
# gri = GridSearchCV(rfc, param_grid, cv=10)
# gri.fit(x, y)
# print(gri.best_params_)
# print(gri.best_score_)

param_grid = {'min_samples_split': np.arange(2, 2+20, 1)}
rfc = RandomForestClassifier(n_estimators=45, random_state=90)
gri = GridSearchCV(rfc, param_grid, cv=10)
gri.fit(x, y)
print(gri.best_params_)
print(gri.best_score_)

param_grid = {'criterion': ['gini', 'entropy']}
rfc = RandomForestClassifier(n_estimators=45, random_state=90)
gri = GridSearchCV(rfc, param_grid, cv=10)
gri.fit(x, y)
print(gri.best_params_)
print(gri.best_score_)
# {'min_samples_split': 2}
# 0.9718804920913884
# {'criterion': 'gini'}
# 0.9718804920913884

#  feature selection
# 方差过滤
# from sklearn.feature_selection import VarianceThreshold
# # print(data.shape, x.shape, y.shape)  # (891, 9) (891, 8) (891,)
# x_selector_1 = VarianceThreshold().fit_transform(x)
# print(x_selector_1.shape)  # (891, 8)
# x_selector_2 = VarianceThreshold(np.median(x.var().values)).fit_transform(x)
# print(x_selector_2.shape)  # (891, 4)

# 卡方检验  是否独立于标签
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# chivalue, p_value = chi2(x, y)
# print(chivalue)
# print('#'*20)
# print(p_value)
# print(p_value < 0.05)
# # k 的选择
# k = chivalue.shape[0] - (p_value > 0.05).sum()
# print(k)
# print(x.shape)
# x_selector = SelectKBest(chi2, k=k).fit_transform(x, y)
# print(x_selector.shape)

# f检验  寻找线性关系
# from sklearn.feature_selection import f_classif
# f, p_value_f = f_classif(x, y)
# k = f.shape[0] - (p_value_f > 0.05).sum()
# print(k)
# print(x.shape)
# x_selector = SelectKBest(f_classif, k=k).fit_transform(x, y)
# print(x_selector.shape)
# 6
# (891, 8)
# (891, 6)

# 互信息法 返回特征与标签的互信息量，0 式没有，1表示完全相关
from sklearn.feature_selection import mutual_info_classif as MIC
# print(MIC(x, y))
# k = MIC(x, y).shape[0] - (MIC(x, y) < 0).sum()
# print(k)
# print(x.shape)
# x_selector = SelectKBest(MIC, k=k).fit_transform(x, y)
# print(x_selector.shape)
# 8
# (891, 8)
# (891, 8)

# 包装法  算法和特征选择同时进行， 计算偏慢，不太适合大型数据
from sklearn.feature_selection import RFE
# clf = DecisionTreeClassifier()
# selector = RFE(clf, n_features_to_select=4, step=1).fit(x, y)
# print(selector.support_.sum())
# print(x.columns)
# print(selector.ranking_)

# 嵌入法  使用算法来选择特征  耗时   当prefit式 TRUE时，不能与 cross_val_score, gridsearchcv一起使用
# from sklearn.feature_selection import SelectFromModel
# clf = DecisionTreeClassifier()
# selector = SelectFromModel(clf, threshold=0.05).fit_transform(x, y)
# print(selector.shape)
# (891, 5)



