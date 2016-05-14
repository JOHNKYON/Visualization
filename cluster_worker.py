# -*- coding: utf-8 -*-

import src
import codecs
import conf

# 读入文件
# input_file = codecs.open("data/source_without_header.txt", 'rb', encoding='utf8')
# raw = input_file.read()

# 分词
# conf.jieba_conf.init()
raw = src.pg.pg_select()

# TODO:raw此时为dict，包含标签和原始数据
# TODO:需要重写init

# text_list = src.init.lsi_init(raw)

# 生成文档向量
dic_corpus = src.lsi.digitalize(text_list)
dictionary = dic_corpus[0]
corpus = dic_corpus[1]

# 用tfidf训练
corpus_tfidf = src.lsi.build_tfidf(corpus)

# 训练lsi模型
lsi = src.lsi.build_lsi(corpus_tfidf, dictionary)

# 获取每个major的bucket_id及匹配度
bucket = src.lsi.topic_cluster(lsi, corpus_tfidf)

src.pg.pg_insert(bucket)

# 获取索引
index = src.lsi.lsi_index(lsi, corpus)

# 用索引和语料库生成关联度矩阵
sim_matrix = src.lsi.sim_matrix(index)

# 聚类结果查询
src.querry.form_bucket(sim_matrix)

# 储存关联度矩阵
output_file = codecs.open("data/sim_matrix.txt", 'wb', encoding='utf8')

for ele in sim_matrix:
    for m in ele:
        output_file.write('('+str(m[0])+',\t'+str(m[1])+')\t')
    output_file.write('\n')

output_file.close()
input_file.close()
