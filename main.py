import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model, learning_curve, model_selection
from sklearn.metrics import precision_score


def set_missing_ages(df):
    """
    使用随机森林填补age空缺
    :param df: 空缺数据集
    :return: 填补后的数据集，随机森林模型
    """
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]  # 用于拟合年龄的字段

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    # 输出
    y = known_age[:, 0]

    # 输入特征
    X = known_age[:, 1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=1000, n_jobs=-1)
    '''
    参数：
        - random_state 
            让结果容易复现。在参数和训练数据不变的情况下，一个确定的随机值将会产生相同的结果
        - n_estimators 子模型数量
        - n_jobs 处理限制
            - -1: 不限制
            -  n: n个处理器
         
    '''
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predicted_ages = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predicted_ages

    return df, rfr


def set_cabin(df):
    """
    处理cabin
    :param df:
    :return:
    """
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


def numerical(df):
    """
    数值化非数值字段
    :param df:
    :return:
    """
    dummies_Cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(df['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')

    df = pd.concat([df, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    return df


def scaling(df):
    """
    缩放Age和Fare
    :param df:
    :return:
    """
    scaler = preprocessing.StandardScaler()

    Age = df.Age.values.reshape(-1, 1)  # fit需要二维向量
    Fare = df.Fare.values.reshape(-1, 1)

    age_scale_param = scaler.fit(Age)
    df['Age_scaled'] = scaler.fit_transform(Age, age_scale_param)
    fare_scale_param = scaler.fit(Fare)
    df['Fare_scaled'] = scaler.fit_transform(Fare, fare_scale_param)

    return df


def preprocessor(df):
    """
    预处理数据集
    :param df:
    :return:
    """
    df, rfr = set_missing_ages(df)
    df = set_cabin(df)
    df = numerical(df)
    df = scaling(df)

    return df


def train(df):
    """
    训练模型
    :param df:
    :return:
    """
    # 用正则取出我们要的属性值
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.values

    # y即Survival结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]

    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)

    return clf


# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
#                         train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
#     """
#     画出data在某模型上的learning curve.
#
#     :param
#     estimator : 你用的分类器。
#     title : 表格的标题。
#     X : 输入的feature，numpy类型
#     y : 输入的target vector
#     ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
#     cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
#     n_jobs : 并行的的任务数(默认1)
#     """
#     train_sizes, train_scores, test_scores = model_selection.learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
#
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#
#     if plot:
#         plt.figure()
#         plt.title(title)
#         if ylim is not None:
#             plt.ylim(*ylim)
#         plt.xlabel(u"训练样本数")
#         plt.ylabel(u"得分")
#         plt.gca().invert_yaxis()
#         plt.grid()
#
#         plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
#                          alpha=0.1, color="b")
#         plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
#                          alpha=0.1, color="r")
#         plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
#         plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")
#
#         plt.legend(loc="best")
#
#         plt.draw()
#         plt.show()
#         plt.gca().invert_yaxis()
#
#     midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
#     diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
#     return midpoint, diff


def bagging_train(df, show_learning_curve=False):
    """
    使用Bagging优化
    :param df:
    :return:
    """
    train_df = df.filter(
        regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
    train_np = train_df.as_matrix()

    # y即Survival结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]

    # fit到BaggingRegressor之中
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True,
                                   bootstrap_features=False, n_jobs=-1)
    bagging_clf.fit(X, y)

    # if show_learning_curve:
    #     plot_learning_curve(bagging_clf, 'Bagging', X, y)

    return bagging_clf


def read_train():
    return pd.read_csv('data/train.csv')


def read_test():
    return pd.read_csv('data/test.csv')


if __name__ == '__main__':
    train_df = preprocessor(read_train())
    bagging_clf = bagging_train(train_df)
    clf = train(train_df)

    # 预测
    test_df = preprocessor(read_test())
    test = test_df.filter(
        regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
    predictions = bagging_clf.predict(test)

    result = pd.DataFrame({'PassengerId': test_df['PassengerId'].values, 'Survived': predictions.astype(np.int32)})
    result.to_csv("data/logistic_regression_bagging_predictions.csv", index=False)

    all_data = train_df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    X = all_data.values[:, 1:]
    y = all_data.values[:, 0]
    model_selection.cross_val_score(clf, X, y, cv=5)