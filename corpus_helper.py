__author__ = 'jellyzhang'
from MysqlHelper import *
import hashlib



def  insert_data(porn_file,unporn_file):
    mysql_helper = MysqlHelper('193.168.15.136', 'test', 'test', 'porn', 'utf8')
    with open(porn_file,'r',encoding='utf-8',errors='ignore') as fporn_read:
        for line in fporn_read:
            sql_query="select * from porn where Content= %s"
            param_query=line.rstrip()
            result=mysql_helper.find(sql_query,param_query)
            if result==0:
                sql_insert='insert into porn(ID,Content) values (%s,%s)'
                md5 = hashlib.md5()
                md5.update(line.rstrip().encode(encoding='utf-8'))
                param_insert=md5.hexdigest(),line.rstrip()
                mysql_helper.cud(sql_insert,param_insert)
    with open(unporn_file, 'r', encoding='utf-8') as funporn_read:
        for line in funporn_read:
            sql_query = "select * from unporn where Content= %s"
            param_query = line.rstrip()
            result = mysql_helper.find(sql_query, param_query)
            if result == 0:
                sql_insert = 'insert into unporn(ID,Content) values (%s,%s)'
                md5 = hashlib.md5()
                md5.update(line.rstrip().encode(encoding='utf-8'))
                param_insert = md5.hexdigest(), line.rstrip()
                mysql_helper.cud(sql_insert, param_insert)

def export_corpus():
    mysql_helper = MysqlHelper('193.168.15.136', 'test', 'test', 'porn', 'utf8')
    with open('data/porn.txt', 'w', encoding='utf-8', errors='ignore') as fporn_write:
        result_porn=mysql_helper.exeQuery('select Content from porn')
        for row in result_porn._rows:
            fporn_write.write('{}\n'.format(row[0]))
    with open('data/unporn.txt', 'w', encoding='utf-8') as funporn_write:
        result_unporn = mysql_helper.exeQuery('select Content from unporn')
        for row in result_unporn._rows:
            funporn_write.write('{}\n'.format(row[0]))


if __name__=='__main__':
    insert_data('data/import_porn.txt','data/import_unporn.txt')