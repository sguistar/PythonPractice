import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By





















if __name__ == '__main__':
    driver = webdriver.Edge()
    url = 'https://spa5.scrape.center/'
    driver.get(url)
    time.sleep(5)
    for page in range(1, 3):
        for i in range(1, 19):
            book_img_xpath = f'//*[@id="index"]/div[1]/div/div/div[{i}]/div/div/div[1]/div/a/img' # 书籍图片
            book_arthor_xpath = f'//*[@id="index"]/div[1]/div/div/div[{i}]/div/div/div[2]/div/p' # 书籍作者
            book_name_xpath = f'//*[@id="index"]/div[1]/div/div/div[{i}]/div/div/div[2]/div[1]/a' # 书籍名称
            book_link_xpath = f'//*[@id="index"]/div[1]/div/div/div[{i}]/div/div/div[2]/div[1]/a' # 书籍链接
            book_link = driver.find_element(By.XPATH, book_link_xpath).get_attribute('href')
            print(f'book_link:{book_link}')
            # book_arthor = driver.find_element(By.XPATH, book_arthor_xpath).text if book_arthor_xpath is not None else '无作者信息'
            if driver.find_elements(By.XPATH, book_arthor_xpath):
                book_arthor = driver.find_element(By.XPATH, book_arthor_xpath).text
            else:
                book_arthor = 'None'
            print(f'book_arthor:{book_arthor}')
            book_name = driver.find_element(By.XPATH, book_name_xpath).text
            print(f'book_name:{book_name}')
            book_img_url = driver.find_element(By.XPATH, book_img_xpath).get_attribute('src')
            print(f'book_img_url:{book_img_url}')
            driver.get(book_link)
            time.sleep(10)
            #rating = driver.find_element(By.XPATH, '//*[@id="detail"]/div[2]/div/div[1]/div[2]/div[2]/div').text
            if driver.find_elements(By.XPATH, '//*[@id="detail"]/div[1]/div/div/div[1]/div/div[1]/a/span'):
                rating = driver.find_element(By.XPATH, '//*[@id="detail"]/div[1]/div/div/div[1]/div/div[1]/a/span').text
            else:
                rating = 'None'
            print(f'rating:{rating}')
            
            if driver.find_elements(By.XPATH, '//*[@id="detail"]/div[1]/div/div/div[1]/div/div[1]/p'):
                book_intro = driver.find_element(By.XPATH, '//*[@id="detail"]/div[1]/div/div/div[1]/div/div[1]/p').text
            else:
                book_intro = 'None'
            print(f'book_intro:{book_intro}')
            
            if driver.find_elements(By.XPATH, '//*[@id="detail"]/div[1]/div/div/div[1]/div/div[1]/div[1]'):
                book_tags = driver.find_element(By.XPATH, '//*[@id="detail"]/div[1]/div/div/div[1]/div/div[1]/div[1]').text
            else:
                book_tags = 'None'
            print(f'book_tags:{book_tags}')
            
            if driver.find_elements(By.XPATH, '//*[@id="detail"]/div[1]/div/div/div[1]/div/div[1]/div[2]/p[1]'):
                book_price = driver.find_element(By.XPATH, '//*[@id="detail"]/div[1]/div/div/div[1]/div/div[1]/div[2]/p[1]').text
            else:
                book_price = 'None'
            print(f'book_price:{book_price}')
            
            if driver.find_elements(By.XPATH, '//*[@id="detail"]/div[1]/div/div/div[1]/div/div[1]/div[2]/p[3]'):
                published_time = driver.find_element(By.XPATH, '//*[@id="detail"]/div[1]/div/div/div[1]/div/div[1]/div[2]/p[3]').text
            else:
                published_time = 'None'
            print(f'published_time:{published_time}')
            
            if driver.find_elements(By.XPATH, '//*[@id="detail"]/div[1]/div/div/div[1]/div/div[1]/div[2]/p[4]'):
                publisher = driver.find_element(By.XPATH, '//*[@id="detail"]/div[1]/div/div/div[1]/div/div[1]/div[2]/p[4]').text
            else:
                publisher = 'None'
            print(f'publisher:{publisher}')
            
            if driver.find_elements(By.XPATH, '//*[@id="detail"]/div[1]/div/div/div[1]/div/div[1]/div[2]/p[5]'):
                total_pages = driver.find_element(By.XPATH, '//*[@id="detail"]/div[1]/div/div/div[1]/div/div[1]/div[2]/p[5]').text
            else:
                total_pages = 'None'
            print(f'total_pages:{total_pages}')
            
            if driver.find_elements(By.XPATH, '//*[@id="detail"]/div[1]/div/div/div[1]/div/div[1]/div[2]/p[6]'):
                ISBM = driver.find_element(By.XPATH, '//*[@id="detail"]/div[1]/div/div/div[1]/div/div[1]/div[2]/p[6]').text
            else:
                ISBM = 'None'
            print(f'ISBM:{ISBM}')
            
            with open('books.txt', 'a', encoding='utf-8') as f:
                f.write(f'{book_arthor},{book_name},{book_link},{book_img_url},{rating},{book_intro},{book_tags},{book_price},{published_time},{publisher},{total_pages},{ISBM}\n')
            
            driver.back()
            time.sleep(10)
            
            
        # 点击下一页
        next_page_xpath = f'//*[@id="index"]/div[2]/div/div/div/button[2]/i'
        next_page = driver.find_element(By.XPATH, next_page_xpath)
        next_page.click()
        time.sleep(5)
    
    
    