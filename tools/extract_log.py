# coding=utf-8
# 该文件用来提取训练log，去除不可解析的log后使log文件格式化，生成新的log文件供可视化工具绘图

import random

f = open('paul_train_log.txt')
train_log = open('paul_train_log_new.txt', 'w')

for line in f:
    # 去除多gpu的同步log
    if 'Syncing' in line:
        continue
    # 去除除零错误的log
    if 'nan' in line:
        continue
    train_log.write(line)

f.close()
train_log.close()
