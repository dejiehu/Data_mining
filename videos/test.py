from datetime import datetime

start = datetime.now()
print(start)


for i in range(0,50000000):
    a = 10

end = datetime.now()
print(end - start)