import yaml
import os
import argparse


class ChangeJDETrackerThresh:
    def __init__(self, yaml_path="configs/mot/fairmot/_base_/fairmot_dla34.yml", thresh=0.4):

        self.path = os.getcwd()
        self.yaml_path = self.path+"/"+yaml_path
        print(self.yaml_path)
        self.thresh = thresh
        self.change_JDETracker_thresh()

    def change_JDETracker_thresh(self):
        # 打开yaml文件
        print(os.getcwd())
        print("***获取yaml文件数据***")
        with open(self.yaml_path, "r+", encoding="utf-8") as file:
            file_data = file.read()
            file.close()
            print(file_data)
            print("类型：", type(file_data))
            with open(self.yaml_path, "w+", encoding="utf-8") as file_w:
                # 将字符串转化为字典或列表
                print("***转化yaml数据为字典或列表***")
                data = yaml.load(file_data)
                print(data)
                data['JDETracker']['conf_thres'] = self.thresh
                print(data['JDETracker']['conf_thres'])
                yaml.dump(data, file_w)
                file_w.close
        with open(self.yaml_path, "r+", encoding="utf-8") as file:
            file_data = file.read()
            file.close()
            data = yaml.load(file_data)
            print(data['JDETracker']['conf_thres'])
        # print("类型：", type(data))
        return data
