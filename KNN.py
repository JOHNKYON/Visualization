# -*- coding:utf-8 -*-
import numpy as np
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import conf
import data
import src

__author__ = 'JOHNKYON'

conf.jieba_conf.init()

pg_conf = conf.pg_config

raw = src.pg.pg_select(pg_conf)

raw = [[x[0]+x[1], x[2]] for x in raw]

x = [[a[0]] for a in raw]

x = src.init.class_init_tf_idf(x)

# for doc in x:
#     print doc

# y = [[(a[1]-10400000)/1000] for a in raw]
#
# x = np.array(x)
# labels = np.array(y)
#
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
#
# clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
# clf.fit(x_train, y_train)
#
# precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
#
# answer = clf.predict_proba(x)[:, 1]
# print(classification_report(y, answer, target_names=y))
