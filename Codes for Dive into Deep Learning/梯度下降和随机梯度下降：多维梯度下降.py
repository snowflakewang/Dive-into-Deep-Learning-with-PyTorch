import d2lzh_pytorch.utils as d2l
import math
import numpy as np

'''以下是梯度下降算法'''
def train_2d(trainer):
    x1,x2,s1,s2=-5,-2,0,0#s1,s2是自变量的状态
    results=[(x1,x2)]
    for i in range(20):
        x1,x2,s1,s2=trainer(x1,x2,s1,s2)
        results.append((x1,x2))
        print('epoch %d,x1 %f,x2 %f'%(i+1,x1,x2))
    return results

def show_trace_2d(f, results):
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')

def f_2d(x1,x2):
    return x1**2+2*x2**2

def gd_2d(x1,x2,s1,s2):
    return (x1-2*eta*x1,x2-4*eta*x2,0,0)

eta=0.1
show_trace_2d(f_2d,train_2d(gd_2d))

'''以下是随机梯度下降算法'''
def sgd_2d(x1,x2,s1,s2):
    return (x1-eta*(2*x1+np.random.normal(0,1)),x2-eta*(4*x2+np.random.normal(0,1)),0,0)
    #用mean=0，var=1的正态分布引入扰动来模拟随机梯度下降中可能会出现的扰动
eta=0.1
show_trace_2d(f_2d,train_2d(sgd_2d))

