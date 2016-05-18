# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


__author__ = "JOHNKYON"


def imamge_2D(points, label):
    """
    用于生成二维图像
    :param points:
    :return:
    """

    plt.scatter(points[:, 0], points[:, 1], c=label)
    plt.savefig('images/nlp-generated-test.png')
