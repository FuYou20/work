# require modelscope>=0.3.7，目前默认大于0.3.7，您检查确认一下即可
# 按照更新镜像的方法处理或者下面的方法
# pip install --upgrade modelscope -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
# 需要单独安装`decord`，安装方法：pip install `decord
import base64
import io
import json
from io import BytesIO

import chardet
import numpy as np
import torch
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from PIL import Image
from NpEncoder import NpEncoder
from read_img import ImageReader
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

pipeline = pipeline(task=Tasks.multi_modal_embedding,
                        model='damo/multi-modal_clip-vit-base-patch16_zh', model_revision='v1.0.1')
image_list = ''
img_embedding = ''
def get_match_rate(item):
    return item["matchRate"]
# 加载图片数据函数
def load_data():
    reader = ImageReader('imgs')
    global image_list
    image_list = reader.read_images()

    # 支持一张图片(PIL.Image)或多张图片(List[PIL.Image])输入，输出归一化特征向量
    img_embedding = pipeline.forward({'img': image_list})['img_embedding']  # 2D Tensor, [图片数, 特征维度]

    return img_embedding

# 依据下标返回数据
@app.route('/getImg',methods=['GET'])
def getImg():
    data = request.json
    index = data.get('index')
    print(index)


@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/textToImage',methods=['POST'])
def textToImage():
    data = request.json
    text_input = data.get('textInput')
    img_count =  data.get('imgCount')
    res=match(text_input,img_count)
    return res;
@app.route('/imageToImage',methods=['POST'])
def imgToImg():
    # 接收图片
    upload_img = request.files['file']
    img_count = request.form.get('imgCount')
    byteImg = io.BytesIO(upload_img.stream.read())

    pilImg = Image.open(byteImg)

    # 获取图片名
    file_name = upload_img.filename
    print(type(pilImg))
    return match2(pilImg,img_count)

    # 计算图文相似度
def match(text_input,img_count):
    with torch.no_grad():
        text_embedding = pipeline.forward({'text': text_input})['text_embedding']  # 2D Tensor, [文本数, 特征维度]
        # 计算内积得到logit，考虑模型temperature
        logits_per_image = (text_embedding / pipeline.model.temperature) @ img_embedding.t()
        # 根据logit计算概率分布
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print("图文匹配概率:", probs)
        print(img_count)
        maxIndex = np.argsort(probs)
        maxIndex=maxIndex[0][-int(img_count):]

        print(maxIndex)
        inner_array = probs[0]
        #返回匹配度最高的图片数组下标以及匹配度
        res_list = []
        buffer =BytesIO()
        for index in maxIndex:
            img = image_list[index]
            print(img)
            img.save(buffer, format='JPEG')
            img_bytes = buffer.getvalue()
            buffer.truncate(0)
            buffer.seek(0)
            image_base64 = base64.b64encode(img_bytes).decode('utf-8')

            res_list.append({"base64":image_base64,"matchRate":inner_array[index]})

        res_list = sorted(res_list, key=get_match_rate, reverse=True)
        return json.dumps([res_list,inner_array[maxIndex]],cls=NpEncoder)

def match2(img_input,img_count):
    with torch.no_grad():
        img_embedding2 = pipeline.forward({'img': img_input})['img_embedding']  # 2D Tensor, [图片数, 特征维度]
        logits_per_image = (img_embedding2 / pipeline.model.temperature) @ img_embedding.t()
        # 根据logit计算概率分布
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print("图图匹配概率:", probs)
        print(img_count)
        maxIndex = np.argsort(probs)
        maxIndex = maxIndex[0][-int(img_count):]

        print(maxIndex)
        inner_array = probs[0]
        # 返回匹配度最高的图片数组下标以及匹配度
        res_list = []
        buffer = BytesIO()
        for index in maxIndex:
            img = image_list[index]
            print(img)
            img.save(buffer, format='JPEG')
            img_bytes = buffer.getvalue()
            buffer.truncate(0)
            buffer.seek(0)
            image_base64 = base64.b64encode(img_bytes).decode('utf-8')

            res_list.append({"base64": image_base64, "matchRate": inner_array[index]})

        res_list = sorted(res_list, key=get_match_rate, reverse=True)
        return json.dumps([res_list, inner_array[maxIndex]], cls=NpEncoder)
if __name__ == '__main__':
    img_embedding=load_data()
    app.run()