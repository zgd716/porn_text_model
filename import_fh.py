from multiprocessing.dummy import Pool
import requests
from  bs4 import BeautifulSoup
import re
"""
番号写到txt中
"""


def get_fh(url):
    content = requests.get(url).content
    soup = BeautifulSoup(content, 'html.parser')
    with open('fh.txt','a',encoding='utf-8')as fwrite:
        for div in soup.find_all('td'):
            if re.match('^[a-zA-Z][-.a-zA-Z0-9:_]*$',div.text):
                fwrite.write('{}\n'.format(div.text))
            #print(div.contents[2].string)


"""
获取番号前缀
"""
def get_fh_prefix(file,out_file):
    prefixs=set()
    with open(file,'r')as fread,open(out_file,'w')as fwrite:
        for line in fread:
            prefixs.add(line.rstrip().split('-')[0])
        for prefix in prefixs:
            fwrite.write('{}\n'.format(prefix))


if __name__=='__main__':
    '''
    爬取网页番号信息
    '''
    # urls=[]
    # for i in range(7431):
    #     urls.append('https://www.wxxyc.com/vod-show-id-1-p-{}.html'.format(i+1))
    #
    # for i in range(len(urls)):
    #     print('page:{}'.format(i+1))
    #     get_fh(urls[i])
    """
    统计出番号前缀
    """
    get_fh_prefix('data/fh.txt','data/fh_prefix.txt')