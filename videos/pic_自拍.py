import datetime
import os
import requests #用于请求网页
import re  #正则表达式，用于解析筛选网页中的信息

from lxml import etree

image = '自拍'
if not os.path.exists(image):
    os.mkdir(image)
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'

}


# response = requests.get('https://yingpian8.com/2023/10/10',headers=headers)
response = requests.get('https://www.28rty.com/e/action/ShowInfo.php?classid=1&id=66525',headers=headers)
# start = datetime.now()
# print(start)
response.encoding = 'utf-8'
print("response.status_code",response.status_code)
page_text = response.text
# print(page_text)
    #数据解析：src属性值 alt属性值
tree = etree.HTML(page_text)
# print(tree)
li_list = tree.xpath('//div[@class="entry"]/ignore_js_op')  #会匹配所有具有类名为"entry"的div元素，然后在这些div元素内寻找ignore_js_op元素
print(len(li_list), li_list)
i  = 0
# print("1",li_list[0].xpath('./@href')[0])
# print("2",li_list[0].xpath('./img/@src')[0])
for li in li_list:

    img_src = li.xpath('./img/@src')[0]
    print(img_src)

    s = img_src.split('/')[-1]  # 截取图片后缀，得到表情包格式，如jpg ，gif


    res = requests.get(url= img_src,headers = headers)
    print(res.status_code)

    with open(image + '/'  + s , mode='wb') as file:
        file.write(res.content)
        i += 1
        print( i,"/",len(li_list) , ":" + image + '/'   + s )
