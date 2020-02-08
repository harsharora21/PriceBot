import math
import numpy as np

def signum(x):
    if x==0:
        return 0.0
    return math.copysign(1,x)

def convTimeSeries(f,n):
	x=[]
	y=[]
	for i in range(0,len(f) - n):
		x.append(f[i:(i+n)])
		y.append([f[i+n]])
	return x,y

def convToClassification(x,y):
	n=len(x[0])
	out=[]
	for i in range(0,len(y)):
		out.append([y[i][0]-x[i][n-1]])
	return out

def createDiffArray(arr):
    out=[]
    for i in range(0,len(arr)-1):
        out.append(arr[i+1]-arr[i])
    return out

def normalize(x,mmax,mmin):
    out=(x-mmin)/(mmax-mmin)
    return out

def denormalize(x,mmax,mmin):
    out=(x*(mmax-mmin)) + mmin
    return out

def checkDir(corr,pre,ths):
    ctr=0
    tot=0
    for i in range(1,len(corr)):
        if(abs(corr[i-1]-pre[i])<ths):
            tot+=1
            ctr+=int((corr[i-1]<corr[i])==(corr[i-1]<pre[i]))
    print((100*ctr)/(1.0*tot))
    return ctr,tot

def checkBoundsStrat(corr,pre):
    ctr=0
    tot=0
    money=0
    for i in range(1,len(corr)):
        if corr[i-1]>pre[i]:
            tot+=1
            ctr+=int(corr[i-1]>corr[i])
            money+=corr[i-1]-corr[i]
    print(money)
    return ctr,tot

