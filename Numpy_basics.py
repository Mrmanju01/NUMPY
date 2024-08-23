import numpy as np

a=np.array([1,2,3])
print(a)
print(a[0])
print(a[1])
print(a[2])

import time
import sys
#diff b/w numpy and list
b=range(1000)
print(sys.getsizeof(5)*len(b)  
#op-28000

c=np.arange(1000)
print(c.size*c.itemsize)
#op-4800

**********************
#time saving
size=100000
l1=range(size)
l2=range(size)
a1=np.arange(size)
a2=np.arange(size)

start=time.time()
re=[(x+y) for x,y in zip(l1,l2)]
print(re)
print((time.time()-start)*1000)

start=time.time()
re=a1+a2
print(re)
print((time.time()-start)*1000)

***********************
BASICS OF NUMPY

a=np.array([1,2],[3,4],[5,6]])
print(a)
print(a.ndim)
print(a.itemsize)
print(a.shape)


a=np.array([1,2],[3,4],[5,6]],dtype=np.float(64))

print(a)
print(a.shape)
print(a.itemsize)

**************
		ARRAYS
np.zeros((3,4))
np.ones((3,4))
np.eye((3,4))
np.full((2,4),20)

n=np.arange(1,10,1)
print(n)

#concatenation example
print(np.char.add(["hello","hi"],["manju","reddy"])

print(np.char.multiply("hello',3)

print(np.char.multiply("hello',30,fillchar="*")


print(np.char.capitalize("hello')


print(np.char.title("hello mr manjunatha reddy")

print(np.char.lower("mANJU"))
print(np.char.upper("manjunathareddy"))

print(np.char.split("hey hello all"))
print(np.char.splitlines("am i good boy?\n yes you are a boy"))

print(np.char.strip(["hey","hello"],"h"))

print(np.char.join([":","-"],["dmy","ymd"]))


s="maam"
k=s[-1:]
if s==k:
   print("its a reversed string",k)


****************
Array manipulation-changing in shape

import numpy as np

a=np.arange(9)
print("original array:")
print(a)


b=a.reshape(3,3)
print("modified array:")
print(b)
print(b.flattern())------>o/p 0 to 8
print(b.flattern(order="F" or "C" or "A"))------>o/p array[0 to 8]in zigzag

a=np.arange(12).reshape(4,3)
print(a)
print(np.transpose(a))

b=np.arange(8)

print(b)
c=b.reshape(2,2,2)
print(c)

print(np.rollaxis(c,2,1))
print(np.swapaxes(c,1,2))
********************
	NUMPY ARITHEMATIC OPERATIONS
import numpy as np
a=np.arange(9).reshape(3,3)
print(a)
b=np.array([10,11,12])
print(b)

np.add(a,b)
np.subtract(a,b)
np.multiply(a,b)
np.divide(a,b)

***SLICING***
a=np.arange(20)
print(a)
a[4:]
a[:4]
s=slice(2,9,2)
a[s]
******iterating*******

a=np.arange(0,45,5).reshape(3,3)
for x in np.nditer(a):
	print(x)
C-STYLE AND F-STYLE
print(a)
for x in np.nditer(a,order="C"):
	print(x)

for x in np.nditer(a,order="A"):
	print(x)

********JOINING ARRAYS*********
a=np.array([[1,2],[3,4]])
b=np.array([[5,6],[7,8]])


print(np.concate(a,b),axis=1)
print(np.concate(a,b),axis=0)

*******SPLITTING*******
a=np.arange(10)
print(np.split(a,4))
print(np.split(a,[2,5]))

*******RESIZING ARRAY**********
a=np.array([[1,2,3],[4,5,6]])
print(a)
print(a.shape)
b=np.resize(a,(3,2))
print(b)
print(b.shape)


*********NUMPY HISTOGRAM***************
from matplot import pyplot as pt
a= np.array([20,87,4,40,53,74,56,51,11,20,40,15,79,25,27])
pt.hist(a,bins=[0,20,40,60,80,100])
pt.title("histogram")
pt.show()

************USEFUL FUNCTIONS**********
a=np.linespace(1,3,10)
print(a)

a=np.array([1,3,10],[3,4,5])
print(a.sum(axis=0))
print(np.sqrt(a))
print(np.std(a))

x=np.array([1,2,3],[4,5,6])
print(x.ravel())
a=np.array([1,2,3])

print(np.log10(a))


