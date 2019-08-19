from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

from IPython import display

import numpy as np
import pandas as pd

class GA():
  def __init__(self, nums, bound, func, DNA_SIZE=None, cross_rate=0.8, mutation=0.003):
    pass
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
