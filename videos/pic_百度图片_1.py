import os
import requests #用于请求网页
import re  #正则表达式，用于解析筛选网页中的信息

from lxml import etree

image = 'baidu_images_test'
if not os.path.exists(image):
    os.mkdir(image)
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'

}


response = requests.get('https://yingpian8.com/2023/10/13',headers=headers)

response.encoding = 'utf-8'
print(response.status_code)
page_text = response.text
    # print(page_text)
    #数据解析：src属性值 alt属性值
tree = etree.HTML(page_text)
li_list = tree.xpath('//div[@class="entry"]/p/a')
print(len(li_list),li_list)
i = 0
for li in li_list:
    if 'JPG' in li.xpath('./@href')[0]:
        img_src = li.xpath('./img/@src')[0] + '/DSC_2032.' + 'JPG'
        img_src = str.replace(img_src, 'jpg', 'JPG')

    else:
        img_src = li.xpath('./img/@src')[0] + '/DSC_2032.' + 'jpg'
    # print(li.xpath('./@href')[0].split('.')[-2])

    s = img_src.split('/')[-2]  # 截取图片后缀，得到表情包格式，如jpg ，gif
    img_src = str.replace(img_src,'th','i')
    # print(img_src)

    res = requests.get(url= img_src,headers = headers)
    print(res.status_code)

    with open(image + '/'  + s , mode='wb') as file:
        file.write(res.content)
        i+=1
        print(i,"/",len(li_list), ":" + image + '/'   + s )
