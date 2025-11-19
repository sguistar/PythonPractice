import requests
from bs4 import BeautifulSoup
import time


def getHtmlText(url):
    h = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36'}
    try:
        resp = requests.get(url, headers=h, timeout=2)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        return resp.text
    except:
        return "error"


def parse_data(html):
    soup = BeautifulSoup(html, 'lxml')
    divs = soup.find_all('div', attrs={'class': 'typecont'})
    for div in divs:
        tags = div.find_all('a', attrs={'target': '_blank'})
        # print(len(tags))
        for tag in tags:
            print(tag.get('href'))


if __name__ == '__main__':
    url = 'https://www.gushiwen.cn/gushi/tangshi.aspx?page=1'
    html = getHtmlText(url)
    data = parse_data(html)
