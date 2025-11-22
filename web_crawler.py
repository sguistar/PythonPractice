from selenium import webdriver
from time import sleep
from selenium.webdriver.common.by import By

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    n = 1
    edgeoption = webdriver.EdgeOptions()
    edgeoption.add_argument('--start-maximized')
    driver=webdriver.Edge(options=edgeoption)
    driver.get('https://www.lanqiao.cn/courses/')
    sleep(3)

    #js='return window.scrollTo(0,document.body.scrollHeight);'
    for i in range(1,33):
        course_xpath = f'//*[@id="__layout"]/div/div[4]/div[1]/div[4]/div[{i}]/a/div[2]'
        course_name=driver.find_element(by=By.XPATH,value=course_xpath)
        course_info_xpath = f'//*[@id="__layout"]/div/div[4]/div[1]/div[4]/div[{i}]/a/div[5]/div[1]'
        course_info=driver.find_element(by=By.XPATH,value=course_info_xpath)
        course_tags_xpath = f'//*[@id="__layout"]/div/div[4]/div[1]/div[4]/div[{i}]/a/div[3]/div'
        course_tags=driver.find_element(by=By.XPATH,value=course_tags_xpath)
        print(course_name.text,course_info.text,'\n',course_tags.text)
        if i % 8 == 0:
            js=f'return window.scrollTo(0,{n*1000});'
            driver.execute_script(js)
            n += 1
            sleep(5)
            
    
    driver.quit()
    
class A(nn.Module):
    f'{__name__}'
def f(self):
    self = 10
    print(self)
