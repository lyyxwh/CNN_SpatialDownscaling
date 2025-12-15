# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:55:39 2020

@author: xiao
下载链接列表格式样例：
http://101.201.177.119/dlorder/orderData/xxx/Z_NAFP_C_BABJ_20191029012028_P_CLDAS_RT_CHN_0P0625_HOR-RSM000010-2019102900.nc?
Expires=xxx&OSSAccessKeyId=xxxxx&Signature=xxx&dataCode=NAFP_CLDAS2.0_RT&userId=xxx
"""


import os
import urllib.request


def getnc(dataList, localPath):

    if not os.path.exists(localPath):  # 新建文件夹
        os.mkdir(localPath)
    with open(dataList,'r') as f:
        lines = f.readlines()
        for line in lines:
            file_name = line.split('?')[0]  # 文件名
            file_name_1 = file_name.split('/')[-1]
            print(file_name_1)
            urllib.request.urlretrieve(line, localPath +'\{}'.format(file_name_1))

            

if __name__ == '__main__':
    dataList = "G:\\cldas\\result.txt" # 下载链接列表
    localPath= "G:\\cldas\\data\\2018"  # 下载数据路径
    getnc(dataList,localPath) # 下载数据
