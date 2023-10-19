import os
import requests #用于请求网页
import re  #正则表达式，用于解析筛选网页中的信息

image = '表情包'
if not os.path.exists(image):
    os.mkdir(image)
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
    }
print("1")
response = requests.get('https://qq.yh31.com/zjbq/',headers=headers)
print("2")
response.encoding = 'utf-8'
print(response.request.headers)
print(response.status_code)
# t = '<img src="(.*?)" alt="(.*?)" width="160" height="120">'
t = '<img src="(.*?)" alt="(.*?)" '
print("response.text",response.text)
result = re.findall(t, response.text)
print(result,"result")
for img in result:
    print(img)
    res = requests.get(img[0])
    res = requests.get(url= img[0],headers = headers)
    print(res.status_code)
    s = img[0].split('.')[-1]  #截取图片后缀，得到表情包格式，如jpg ，gif
    with open(image + '/' + img[1] + '.' + s, mode='wb') as file:
        file.write(res.content)