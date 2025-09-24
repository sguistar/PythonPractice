import requests
from bs4 import BeautifulSoup
import time
import random
import pandas as pd


def getHTMLText(url):  # 抓取页面文本
    try:
        h = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0"
        }
        r = requests.get(url, headers=h, timeout=10)
        print(r.status_code)
        print(r.reason)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        print(
            f"Success fetching the page.\n Start crawling...  \n URL: {r.url} \n Encoding: {r.encoding}"
        )
        return r.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""


def getHTMLImg(url):  # 抓取页面图片
    try:
        h = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0"
        }
        r = requests.get(url, headers=h, timeout=10)
        print(r.status_code)
        print(r.reason)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        print(
            f"Success fetching the page.\n Start crawling...  \n URL: {r.url} \n Encoding: {r.encoding} \n"
        )
        return r.content
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""


def parse(text):  # 解析页面
    pass


def saveData(data):  # 保存数据
    with open("homework.txt", "w", encoding="utf-8") as f:
        f.write(f"{data}\n")


if __name__ == "__main__":
    h = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0"
    }
    url = f"https://movie.douban.com/top250"
    for i in range(0, 226, 25):
        p = {"start": str(i), "filter": ""}
        r = requests.get(url, headers=h, params=p)
        print(r.status_code)
        print(r.reason)
        print(r.request.url)
        time.sleep(random.uniform(1, 3))  # 随机延时，模拟人类行为
        html = r.text
        soup = BeautifulSoup(html, 'html.parser')
        nodes = soup.find_all('div', class_='info')
        for node in nodes:
            print(node.find('span', class_='title').text)
            print('-------------------')
            print(node.find('div', class_='bd').p.text.strip())
            print('-------------------')
            print(node.find('span', class_='rating_num').text if node.find(
                'p', class_='quote') else 'N/A')
            print('-------------------')
            print(node.find('p', class_='quote').span.text if node.find(
                'p', class_='quote') else 'N/A')
            print('-------------------')
            print(node.find('span', property='v:best').find_next_sibling(
                'span').text.replace('人评价', '').strip() if node.find('span') else 'N/A')
            print('-------------------')
        print(f'爬取完成！共获取 {len(nodes)} 部电影信息。')

    # parse(text)
    # saveData(getHTMLText(url))
    # print("Data saved successfully.")
