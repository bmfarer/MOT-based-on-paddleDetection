# import  os
# # os.system("python")
# # os.system("pip show pip")
# # os.system("ssh ")
# cmd='notepad'
# os.system(cmd)
# git 通过 python 脚本上传视频
# 在AIstuidio上面用gitclone把视频克隆下来
import requests
import base64
import json


# 读取文件
def open_file(file_path):
    with open(file_path, 'wb+') as f:
        return f.read()


# 将文件转换为base64编码，上传文件必须将文件以base64格式上传
def file_base64(data):
    data_b64 = base64.b64encode(data).decode('utf-8')
    return data_b64


# 上传文件
def upload_file(file_data):
    file_name = ""  # 文件名
    token = "[token]"
    url = "https://api.github.com/repos/[user]/[repo]/contents/[path]/" + file_name  # 用户名、库名、路径
    headers = {"Authorization": "token " + token}
    content = file_base64(file_data)
    data = {
        "message": "message",
        "committer": {
            "name": "[user]",
            "email": "user@163.com"
        },
        "content": content
    }
    data = json.dumps(data)
    req = requests.put(url=url, data=data, headers=headers)
    req.encoding = "utf-8"
    re_data = json.loads(req.text)
    print(re_data)
    print(re_data['content']['sha'])
    print("https://cdn.jsdelivr.net/gh/[user]/[repo]/[path]" + file_name)


# 在国内默认的down_url可能会无法访问，因此使用CDN访问


if __name__ == '__main__':
    fdata = open_file('77.jpg')
    upload_file(fdata)

