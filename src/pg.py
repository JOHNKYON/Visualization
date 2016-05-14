# -*- coding:utf-8 -*-

import codecs
import json
import psycopg2
from psycopg2.extras import DictCursor

__author__ = "JOHNKYON"


def pg_insert(bucket):
    """
    灌入major_bucket至postgresql
    :param bucket:
    :return:
    """
    input_file = codecs.open("data/major_dict.json", 'rb', encoding='utf8')
    major_list = input_file.read()
    dic = json.loads(major_list)
    bucket = map(lambda x: (dic[str(x[0])], x[1], x[2]), bucket)

    connect = psycopg2.connect(database="dodo", user="data", password="wjf721", host="101.201.183.135", port=3433)

    cursor = connect.cursor()

    for ele in bucket:
        cursor.execute("INSERT INTO major_bucket VALUES(%s, %s, %s)", ele)

    connect.commit()
    cursor.close()
    connect.close()

    input_file.close()


def pg_select():
    """
    查询category和对应的职位描述
    :return dict:
    """
    connect = psycopg2.connect(database="dodo", user="data", password="wjf721", host="101.201.183.135", port=3433)

    cursor = connect.cursor(cursor_factory=DictCursor)

    sql = """   SELECT name, description, category
                FROM company_position_new
                WHERE company_id IS NOT NULL AND category > 100
                ORDER BY category"""

    cursor.execute(sql)

    raw= cursor.fetchall()

    connect.commit()
    cursor.close()
    connect.close()
    return raw