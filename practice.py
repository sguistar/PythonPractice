import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd


def parser_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    trs=soup.find('tbody').children
    # print(len(trs))
    universities = []
    for tr in trs:
        rank = eval(tr.find_all('td')[0].string.strip())  # 排名
        tds = tr.find_all('td')
        name_cn = tds[1].find_all(
            'span', attrs={'class': 'name-cn'})[0].string.strip()  # 中文名称
        name_en = tds[1].find_all(
            'span', attrs={'class': 'name-en'})[0].string.strip()  # 学校类型
        if len(tds[1].find_all('p', attrs={'class': 'tags'})) == 0:
            school_type = ''
        else:
            school_type = tds[1].find_all('p', attrs={'class': 'tags'})[
                0].string.strip()
        area = tds[2].text.strip()
        score = tds[4].text.strip()  # 分数
        un_info = {'排名': rank,
                   '学校中文名': name_cn,
                   '学校英文名': name_en,
                   '学校类型': school_type,
                   '区域': area,
                   '分数': score
                   }
        universities.append(un_info)
    return universities


def saveData(data):
    df = pd.DataFrame(
        data, columns=['排名', '学校中文名称', '学校英文名', '学校类型', '区域', '分数'])
    df.to_excel('university.xlsx', index=False)
    print("✅ 数据已保存到 university.xlsx")


url = 'https://www.shanghairanking.cn/rankings/bcur/2020'
driver = webdriver.Edge()
driver.get(url)
# 等待页面加载完毕
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.TAG_NAME, 'tbody')))
print(driver.title)

all_universities = []
for i in range(1, 20):
    print(f'正在爬取第 {i} 页...')
    try:
        # 点击页码
        pagination = driver.find_element(By.CLASS_NAME, "ant-pagination")
        link = pagination.find_element(By.LINK_TEXT, str(i))
        driver.execute_script("arguments[0].click();", link)
    except Exception as e:
        print(f"⚠️ 翻页失败 {i}: {e}")
        continue

    time.sleep(5)  # 给一点缓冲时间
    html = driver.page_source
    data = parser_html(html)
    all_universities.extend(data)

saveData(all_universities)
driver.quit()
