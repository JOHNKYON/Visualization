# -*- coding: utf-8 -*-

import jieba
from data import stopwords
import re
import gensim


def raw_init(raw):
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


def digitalize(raw):
    """
    此函数用于接受双层数组的中文文档，将其数字化为标准词向量
    :param raw: 此参数为一个双层数组，第一层每个元素为单个文档，第二层每个元素为文档中的词
    :return:    输入一个词向量
    """
    # 生成词袋
    dictionary = gensim.corpora.Dictionary(raw)
    # 生成文档向量
    corpus = [dictionary.doc2bow(text) for text in raw]
    return [dictionary, corpus]


def build_lsi(corpus, dictionary):
    """
    :param corpus, dictionary: corpus为一个文档向量,构成为[(0,n),(1,m),```], dictionary为文档的词袋
    :return:    构造完成的lsi模型
    """
    # 建立模型
    # 这个num_topics是拍脑门决定的，具体效果留待调参
    lsi = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=20)
    return lsi


def build_tfidf(raw):
    """
    此函数用于接受一个文档向量并生成tf-idf模型，
    :param raw: 此参数为一个文档向量，构成为[(0,n),(1,m),```]
    :return: 经tf-idf模型训练后的文档向量，结构与参数相同
    """
    tfidf_model = gensim.models.TfidfModel(raw)
    corpus_tfidf = tfidf_model[raw]
    return corpus_tfidf


def lsi_index(lsi, corpus):
    """
    此函数用于为已建立好的lsi建立索引，返回索引
    :param lsi, corpus: lsi模型和语料库
    :return:    索引
    """
    index = gensim.similarities.MatrixSimilarity(lsi[corpus])
    return index


def most_sim_get(index, num, docu):
    """
    此函数用于获取与查询文档相关最近的num个文档
    :param index:
    :param num:
    :return:
    """


def sim_matrix(index):
    """
    此函数用于将此模型中的所有文档转化为矩阵，为某一文档相对于其他文档的相关度。由于实际需求和内存限制，每一文档实际只存最相近的10个文档
    :param index: 已训练完成的lsi的索引
    :return:    返回一个矩阵，为文档间的相关度
    """
    # output_file = codecs.open("data/lsi_matrix.txt", "wb", encoding='utf8')
    # 应当对此矩阵加入文档索引
    index = map(lambda x: map(lambda m, count: [count, m], x, range(0, len(x))), index)
    # 排序
    index = map(lambda x: sorted(x, key=lambda similarity: similarity[1], reverse=True)[0: 10], index)
    return index


def topic_cluster(lsi, corpus):
    """
    此函数用于将多篇文档适用于训练完成的lsi模型，判断其所属的bucket(此处为topic)，并输出到文件
    :param lsi: 已建立完成的lsi模型
    :param corpus: 用于匹配topic的向量化的语料库
    :return: 返回list，结构为(doc_no:topic_no, ...)
    """
    corpus_lsi = lsi[corpus]
    result = list()
    counter = 0
    for doc in corpus_lsi:
        # 此处进行了排序，直接得到了此文档最接近的topic编号和相近度
        doc_topic_most_match = sorted(doc, key=lambda a: a[1], reverse=True)[0]
        row = (counter, doc_topic_most_match[0], doc_topic_most_match[1])
        result.append(row)
        counter += 1
    return result
