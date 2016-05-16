# -*- coding:utf-8 -*-  
import src
import conf

__author__ = "JOHNKYON"

# 分词
conf.jieba_conf.init()
pg_conf = conf.pg_config

raw = src.pg.pg_select(pg_conf)

# TODO:raw此时为dict，包含标签和原始数据
# TODO:需要重写init

mtr = src.init.tSNE_init(raw)

result = src.t_SNE.plot_build(mtr)

src.image_build.imamge_2D(result)

