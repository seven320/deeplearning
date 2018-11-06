#encoding:utf-8

import numpy as np
import pandas as pd

# import

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# print(train.head(10))


#欠損テーブル
def deficit_table(df):
    null_val = df.isnull().sum()
    percent = 100*df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val,percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
    columns = {0:"欠損数",1:"％"})
    return kesson_table_ren_columns

print("train_rawdata\n",deficit_table(train))
print("test_rawdata\n",deficit_table(test))
#-----------前処理---------------
#欠損を埋める
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")
#正規化
train.replace("male",0,inplace=True)
train.replace("female",1,inplace=True)
train["Embarked"].replace("S",0,inplace=True)
train["Embarked"].replace("C",1,inplace=True)
train["Embarked"].replace("Q",2,inplace=True)
# print(train.head(10))

#欠損を埋める
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Embarked"] = test["Embarked"].fillna("S")
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
#正規化
test.replace("male",0,inplace=True)
test.replace("female",1,inplace=True)
test["Embarked"].replace("S",0,inplace=True)
test["Embarked"].replace("C",1,inplace=True)
test["Embarked"].replace("Q",2,inplace=True)

print("train_replacedata\n",deficit_table(train))
print("test_replacedata\n",deficit_table(test))


#-----------解析------------
from sklearn import tree

target = train["Survived"].values
features_one = train[["Pclass","Sex","Age","Fare","Embarked","SibSp"]].values

#決定木作成
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)


#test
test_features = test[["Pclass","Sex","Age","Fare","Embarked","SibSp"]].values
my_prediction = my_tree_one.predict(test_features)

#結果をcsvへ出力
PassengerId = np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

my_solution.to_csv("my_tree_one2.csv", index_label = ["PassengerId"])
