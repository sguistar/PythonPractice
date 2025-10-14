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
        print(r.status_code, r.reason, "->", url)
        r.raise_for_status()
        r.encoding = 'gb2312'
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


def parse(text):
    soup = BeautifulSoup(text, 'lxml')
    # 同时抓白色行和灰色行
    tr_list = soup.find_all('tr', attrs={'bgcolor': ['#f4f4f4', '#ffffff']})
    print(len(tr_list))
    houses = []
    for tr in tr_list:
        tds = tr.find_all('td')
        if len(tds) < 8:
            continue
        a = tds[0].find('a', target="_blank")
        h = {'详细地址': a.get_text(strip=True) if a else None,
             '区域': tds[1].get_text(strip=True),
             '房型': tds[2].get_text(strip=True),
             '户型': tds[3].get_text(strip=True),
             '租金': tds[4].get_text(strip=True),
             '面积(m²)': tds[5].get_text(strip=True),
             '登记时间': tds[6].get_text(strip=True)}
        detail_a = tds[7].find('a')
        if detail_a and detail_a.has_attr('href'):
            h['详细信息'] = 'https://www.lgfdcw.com/cs/' + detail_a['href']
        else:
            h['详细信息'] = None
        houses.append(h)
    return houses


def saveData(data):  # 保存数据
    df = pd.DataFrame(
        data, columns=['详细地址', '区域', '房型', '户型', '租金', '面积(m²)', '登记时间', '详细信息'])
    df['面积(m²)'] = df['面积(m²)'].str.replace('[㎡�O]', '', regex=True)
    df.to_excel('rental houses.xlsx', index=False)


if __name__ == "__main__":
    all_houses = []
    for i in range(1, 31):  # 爬取1-30页
        url = f'https://www.lgfdcw.com/cs/index.php?userid=&infotype=&dq=&fwtype=&hx=&price01=&price02=&pricetype=&fabuday=&addr=&PageNo={i}'
        text = getHTMLText(url)
        time.sleep(random.randint(1, 3))  # 随机休眠1-3秒，防止被封IP
        houses = parse(text)
        all_houses.extend(houses)
    saveData(all_houses)
    print('爬取完成！')
    print('数据保存完成！')
