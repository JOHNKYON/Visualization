# -*- coding:utf-8 -*-  
import src
import conf

__author__ = "JOHNKYON"

# 分词
conf.jieba_conf.init()

print 'split finished'

pg_conf = conf.pg_config

raw = src.pg.pg_select(pg_conf)

print 'pg finished'

mtr = src.init.tSNE_init(raw)

print 'init finished'

result = src.t_SNE.plot_build(mtr)

print 'build finished'

src.image_build.imamge_2D(result)

