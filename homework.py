import requests
import time

url = 'https://movie.douban.com/top250'
# 请求头伪装成浏览器
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36"
}

# 初始字段
params = {'start': 0, 'filter': ''}

with open('homework.txt', 'w', encoding='utf-8') as f:
    for _ in range(10):
        res = requests.get(url, headers=headers, params=params)  # 建立请求
        res.encoding = res.apparent_encoding  # 编码自适应
        print(f'current page: {res.request.url}')
        f.write(res.text + '\n')  # 把html内容写入homework.txt
        params['start'] += 25
        f.flush()
        time.sleep(1)
