# -*- coding:utf-8 -*-  
import re
import jieba
from data import stopwords


__author__ = "JOHNKYON"


def lsi_init(raw):
    """
    去除文本中的停用词，分词
    :param raw:
    :return 分词后的list,list中的元素为2级list,2级list中的元素为词。此时1级list中元素已经是未数字化的词向量:
    """

    # 去除空格符
    raw_without_space = re.sub(' *', '', raw)
    # 将不同的专业分开作list元素
    # 专业号的正则表达式
    major_re = re.compile(u"\d{6}\D")

    '''# 用于测试匹配数量不一致问题
    out_test = codecs.open("data/re_test.txt", 'wb', encoding='utf8')
    re_find = major_re.findall(raw_without_space)
    counter = 0
    for ele in re_find:
        out_test.write(str(counter) + '\t' + ele + '\n')
        counter += 1
    out_test.close()'''

    # 对整个字符串进行切片,分割符为此正则表达
    raw_splited = major_re.split(raw_without_space)[1:]


    # 分词
    # 载入自定义词典
    jieba.load_userdict("data/jieba_dict.txt")
    raw_cut = map(lambda x: jieba.cut(x, cut_all=False), raw_splited)
    # 去除停用词
    raw_without_sw = map(lambda x: filter(lambda y: y not in stopwords, x), raw_cut)
    return raw_without_sw


def tSNE_init(raw):
    """
    针对t-SNE算法进行数据初始化，使数据格式符合scikit-learn的t-SNE模块输入
    暂定使用主题向量对计算初始值
    :param raw:
    :return:
    """
    pass
