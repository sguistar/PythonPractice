import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd


def parser_html(html):
    soup = BeautifulSoup(html, 'lxml')
    trs = soup.find_all('tr', attrs={'bgcolor': ['#FFFFFF']})
    print(len(trs))
    houses = []
    for tr in trs:
        h = {'详细地址': tr.find_all('a', attrs={'target': '_blank'})[0].string,
             '详情链接': "https://www.lgfdcw.com/cs/"+tr.find_all('a', attrs={'target': '_blank'})[0].attrs["href"],
             '房型': tr.find_all('td')[2].string,
             '户型': tr.find_all('td')[3].string,
             '面积(m²)': tr.find_all('td')[4].string[:-1],
             '出售价格': tr.find_all('td')[5].get_text(strip=True),
             '登记时间': tr.find_all('td')[6].string,
             }
        houses.append(h)

    return houses


def saveData(data):
    df = pd.DataFrame(
        data, columns=['详细地址', '详情链接', '房型', '户型', '面积(m²)', '出售价格', '登记时间'])
    # df['面积(m²)'] = df['面积(m²)'].str.replace('[㎡�O]', '', regex=True)
    df.to_excel('sale houses.xlsx', index=False)


url = 'https://www.lgfdcw.com/cs/'
driver = webdriver.Edge()
driver.get(url)
time.sleep(5)
print(driver.title)

all_houses = []
for i in range(1, 11):
    xpath = f'/html/body/table[4]/tbody/tr/td[2]/table[4]/tbody/tr/td[1]/div/div/a[{i}]'
    driver.find_element(By.XPATH, xpath).click()
    html = driver.page_source
    data = parser_html(html)
    all_houses.extend(data)

saveData(all_houses)
driver.quit()
