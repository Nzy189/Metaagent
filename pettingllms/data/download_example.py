#!/usr/bin/env python3
"""
Simple example script to download CodeContests dataset
"""

from download_codecontests import download_codecontests_dataset

if __name__ == '__main__':
    print("开始下载 CodeContests 数据集...")
    
    # 下载训练集到默认目录 pettingllms/data/codecontests
    success = download_codecontests_dataset()
    
    if success:
        print("数据集下载完成！")
        print("文件保存在: pettingllms/data/codecontests/")
    else:
        print("数据集下载失败！")


