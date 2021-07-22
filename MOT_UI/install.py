import os
os.system("python -m pip install paddlepaddle-gpu==2.1.1.post101 -f https://paddlepaddle.org.cn/whl/mkl/stable.html")
os.system("python -m pip install -r requirements.txt")
os.system("python setup.py install")
os.system("unzip fairmot_final.zip")
os.system("rm -rf fairmot_final.zip")
