import os
import requests #用于请求网页
import re  #正则表达式，用于解析筛选网页中的信息

image = 'baidu_images'
if not os.path.exists(image):
    os.mkdir(image)
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'

}

print("1")
response = requests.get('https://yingpian8.com/2023/08/19',headers=headers)
print("2")
response.encoding = 'utf-8'
print(response.status_code)
t = '<img decoding="async" src="(.*?)" border="0">'
t = ' src="(.*?)" '

# print( response.text)
result = re.findall(t, response.text)
print(result,"result")
for img in result:
    s = img.split('/')[-1]  # 截取图片后缀，得到表情包格式，如jpg ，gif
    img = str.replace(img,'th','i')
    print(img + '/DSC_2032.' + s)
    # res = requests.get(img + 'My-Bed-cover-clean.jpg')
    res = requests.get(url= img + '/DSC_2032.' + s,headers = headers)
    print(res.status_code)

    with open(image + '/'  + '.' + s , mode='wb') as file:
        file.write(res.content)
