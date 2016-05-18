# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


__author__ = "JOHNKYON"


def imamge_2D(points):
    """
    用于生成二维图像
    :param points:
    :return:
    """
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.savefig('images/nlp-generated.png')
