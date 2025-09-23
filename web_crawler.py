import requests
import time
import random


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
    for idx in range(1, 11):
        url = f"https://www.lgfdcw.com/cz/index.php?userid=&infotype=&dq=&fwtype=&hx=&price01=&price02=&pricetype=&fabuday=&addr=&PageNo={idx}"
        time.sleep(random.uniform(1, 3))  # 随机延时，模拟人类行为
        # parse(text)
        saveData(getHTMLText(url))
    print("Data saved successfully.")
