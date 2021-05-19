import multiprocessing
print (multiprocessing.cpu_count())


import psutil 
print (psutil.cpu_count(logical = False))
print(psutil.cpu_count(logical = True))
