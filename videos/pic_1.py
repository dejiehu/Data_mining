import requests
from lxml import etree
import os
if __name__ == '__main__':
    url = 'https://pic.netbian.com/4kdongman/'
    #爬取到页面源码数据
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36 Edg/90.0.818.49'
    }
    response = requests.get(url=url,headers=headers)
    #手动设定编码
    # response.encoding = 'utf-8'
    # print(response)
    page_text = response.text
    # print(page_text)
    #数据解析：src属性值 alt属性值
    tree = etree.HTML(page_text)
    li_list = tree.xpath('//div[@class="slist"]/ul/li')
    print("li_list",li_list)
    # #创建一个文件夹
    if not os.path.exists('./picLibs'):
        os.mkdir('./picLibs')
    for li in li_list:
        print(li,li.xpath('./a/img/@src')[0])
        img_src = 'https://pic.netbian.com'+li.xpath('./a/img/@src')[0]
        img_name = li.xpath('./a/img/@alt')[0]+'.jpg'
        #通用处理中文乱码解决方案
        img_name = img_name.encode('iso-8859-1').decode('gbk')
        # print(img_name,img_src)
        #请求图片，进行持久化存储
        img_data = requests.get(url=img_src,headers=headers).content
        img_path = 'picLibs/'+img_name
        with open(img_path,'wb') as fp:
            fp.write(img_data)
            print(img_name,'下载成功！！')
