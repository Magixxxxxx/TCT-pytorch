import torch,sys

import random


for i in range(100):
	time = "2019-8-{} {}:00".format(int(i/12)+1,i%12*2)
	n = 20 + i + random.randint(-20,20)
	print("{},{}".format(time,n))
    
