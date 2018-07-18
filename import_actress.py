from multiprocessing.dummy import Pool
import requests
from  bs4 import BeautifulSoup

"""
女优写到txt中
"""


def get_actressname(url):
    content = requests.get(url).content
    soup = BeautifulSoup(content, 'html.parser')
    with open('actress.txt','a',encoding='utf-8')as fwrite:
        for div in soup.find_all('div', {'class': 'col-xs-6 col-sm-2 placeholder'}):
            fwrite.write('{}\n'.format(div.contents[2].string))
            #print(div.contents[2].string)


if __name__=='__main__':
    urls=[]
    for i in range(1710):
        urls.append('https://www.wxxyc.com/special-show-p-{}.html'.format(i+1))

    for i in range(len(urls)):
        print('page:{}'.format(i+1))
        get_actressname(urls[i])