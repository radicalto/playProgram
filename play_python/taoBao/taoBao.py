# 淘宝秒杀脚本，扫码登录版
'''
2022-12-25 10:09:00.000000
https://detail.tmall.com/item.htm?abbucket=14&id=670079240918&ns=1&spm=a230r.1.14.1.6f3d69402F4frl
'''
import os
from selenium import webdriver
import datetime
import time
from os import path

from selenium.webdriver.common.by import By

driver = webdriver.Edge('C:\Program Files (x86)\Microsoft\Edge\Application\msedgedriver.exe')


def login(url):
    # 打开淘宝登录页，并进行扫码登录
    driver.get("https://www.taobao.com")
    time.sleep(3)
    if driver.find_element(By.XPATH,r"//*[@id='J_SiteNavLogin']/div[1]/div[1]/a[1]"):
        driver.find_element(By.XPATH,r"//*[@id='J_SiteNavLogin']/div[1]/div[1]/a[1]").click()
        print("请在10秒内完成扫码")
        time.sleep(20)
        driver.get(url)

    time.sleep(3)
    if driver.find_element(By.XPATH,
                           r"//*[@id='root']/div/div[2]/div[2]/div[1]/div/div[2]/div[5]/div/div/div[1]/div/div/div[1]/div/span"):
        driver.find_element(By.XPATH,
                            r"//*[@id='root']/div/div[2]/div[2]/div[1]/div/div[2]/div[5]/div/div/div[1]/div/div/div[1]/div/span").click()
        print("选择ok")
    time.sleep(3)
    now = datetime.datetime.now()
    print('login success:', now.strftime('%Y-%m-%d %H:%M:%S'))


def buy(buytime):
    while True:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        # 对比时间，时间到的话就点击结算
        if now >= buytime:
            driver.refresh()
            try:
                # 点击抢购
                if driver.find_element(By.ID,r"J_LinkBuy"):
                    print("速度点击！！！")
                    driver.find_element(By.ID,r"J_LinkBuy").click()
                    time.sleep(0.09)
                    while now >= buytime:
                        try:
                            print("赶紧买！！！")
                            # driver.find_element(By.CLASS_NAME,r'go-btn').click()
                            driver.find_element(By.LINK_TEXT,r'提交订单').click()
                            # time.sleep(0.09)
                            if driver.find_element(By.XPATH,r"//*[@id='payPassword_rsainput']"):
                                pw_input = driver.find_element(by=By.XPATH, value="//*[@id='payPassword_rsainput']")
                                pw_input.send_keys("762093")
                                if driver.find_element(By.ID, r"validateButton"):
                                    driver.find_element(By.ID, r"validateButton").click()
                        except:
                            time.sleep(0.02)
            except:
                time.sleep(0.08)
        print(now)
        time.sleep(0.05)


if __name__ == "__main__":
    times = input("请输入抢购时间：时间格式：2021-12-29 19:45:00.000000")
    # 时间格式："2022-03-19 11:43:00.000000"
    # 测试可以
    # https://detail.tmall.com/item.htm?spm=a230r.1.14.16.6a903f34xN9uol&id=618488552961&ns=1&abbucket=12&skuId=4988554791826
    url = input("请输入抢购地址")
    login(url)
    buy(times)

