from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

from IPython import display

import numpy as np
import pandas as pd

class GA():
  def __init__(self, nums, bound, func, DNA_SIZE=None, cross_rate=0.8, mutation=0.003):
    # save args:
    # bound      取值范围
    # var_len    取值范围大小
    # POP_SIZE   种群大小
    # POP        编码后的种群[[[1,0,1,...],[1,1,1,...],...]]
    #            一维为各个种群, 二维为各个DNA, 三维为碱基对
    # copy_POP   复制的种群用于重置
    # cross_rate DNA交换概率
    # mutation   基因突变概率
    # func       适应度函数

    # nums为二维N * M矩阵, N为进化种群的总数, M为DNA个数(变量的数量)
    # bound为M * 2的二维矩阵, 形如[(min, max), (min, max), ...]
    # func为方法对象
    # DNA_SIZE可指定大小, 为None时将自动指派大小
    nums = np.array(nums)
    bound = np.array(bound)
    self.bound = bound
    if nums.shape[1] != bound.shape[0]:
      raise Exception(f'范围的数量与变量的数量不一致,您有{nums.shape[1]}个变量, 却有{bound.shape[0]}个范围')
    
    for var in nums:
      for index, var_curr in enumerate(var):
        if var_curr < bound[index][0] or var_curr > bound[index][1]:
          raise Exception(f'{var_curr}不在取值范围内')
    
    for min_bound, max_bound in bound:
      if max_bound < min_bound:
        raise Exception(f'({min_bound}, {max_bound})不是合法的区间')
    
    # 所有变量中的最大值及最小值
    min_nums, max_nums = np.array(list(zip(*bound)))
    # var_len为所有变量的取值区间
    self.var_len = var_len = max_nums - min_nums
    # bit为每个变量按整数编码的二进制数
    bits = np.ceil(np.log2(var_len+1))

    if DNA_SIZE == None:
      DNA_SIZE = int(np.max(bits))
    self.DNA_SIZE = DNA_SIZE

    #POP_SIZE为进化的种群数
    self.POP_SIZE = len(nums)
    POP = np.zeros((*nums.shape, DNA_SIZE))
    for i in range(nums.shape[0]):
      for j in range(nums.shape[1]):
        # 编码方式
        num = int(round((nums[i, j] - bound[j][0]) * ((2**DNA_SIZE) / var_len[j])))
        # 转化为前面空0的二进制字符串然后拆分成矩阵
        POP[i, j] = [int(k) for k in ('{0:0'+str(DNA_SIZE)+'b}').format(num)]
    self.POP = POP

    self.copy_POP = POP.copy()
    self.cross_rate = cross_rate
    self.mutation = mutation
    self.func = func
  # 解码DNA
  def translateDNA(self):
    pass
  # 得到适应度
  def get_fitness(self, non_negative=False):
    pass
  # 选择
  def select(self):
    pass
  # DNA交叉
  def crossover(self):
    pass
  # 基因变异
  def mutate(self):
    pass
  # 进化
  def evolution(self):
    pass
  # 重置
  def reset(self):
    pass
  # 打印当前状态日志
  def log(self):
    pass
  # 一维变量作图
  def plot1D(self, iter_time=200):
    pass
