import datetime
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import ToPILImage
import pandas as pd
import os
import progressbar
import multiprocessing as mp
import numpy as np
import csv
from tqdm import tqdm
from lavis.models import load_model
import faiss
import threading
import time
import cv2
import re
import pandas as pd
from multiprocessing import Process
from urllib.parse import urlparse, urlunparse
from openai import OpenAI
import json
from .utils import GPT,Qianwen
from . import Blip2QA

# 根据路径返回开始和结束时间， 根据两个图片路径切分中间的关键帧 ， 返回切帧图片路径
from .utils.videoCutting import get_start_end_by_path, split_video_for_different_user, return_cut_urls_different_user 


def strvector2float(strvector):
    start_pos = strvector.find('[')
    end_pos = strvector.find(']')
    tem = strvector[start_pos:end_pos + 1]
    floatvector = eval(tem)
    return floatvector

def read_csv_features(file_path):
    # 读取CSV文件
    print(f"Loading Data From {file_path} ... ")
    vectors, pic_index = [], []
    with open(file_path, 'r') as csvfile:
        # CSV文件的第一行是列名
        reader = csv.reader(csvfile)
        header = next(reader)  # 读取并忽略第一行
        for row in reader:
            tensor = row[1]  # 获取指定列的值
            id = row[0]
            vector = strvector2float(tensor)
            vector = np.array(vector)
            vector = vector[np.newaxis, :]
            vectors.append(vector)
            pic_index.append(id)
    return vectors, pic_index

def read_csv_caption(file_path):
    # 读取capotion CSV文件
    print(f"Loading Data From {file_path} ... ")
    captions, pic_index = [], []
    with open(file_path, 'r') as csvfile:
        # CSV文件的第一行是列名
        reader = csv.reader(csvfile)
        header = next(reader)  # 读取并忽略第一行
        for row in reader:
            caption = row[1]  # 获取指定列的值
            id = row[0]
            captions.append(caption)
            pic_index.append(id)
    return captions,pic_index

def build_combined_caption_dict(csv_file):
    combined_dict = {}
    # for csv_file in csv_files:
    df = pd.read_csv(csv_file)
        # 确保 'id' 和 'caption' 列存在
    if 'id' in df.columns and 'caption' in df.columns:
            # 合并字典
        combined_dict=dict(zip(df['id'], df['caption']))
    else:
        print(f"Warning: 'id' and 'caption' columns not found in {csv_file}")
    return combined_dict

def get_caption_by_id(caption_dict, id):
    # 根据给定的 ID 查找 caption
    return caption_dict.get(id, "ID not found")

# 读取模型的方法
def load_and_prepare_model(device):
    """加载并准备模型"""
    print(f"Loading Blip2 Model...")
    model = load_model("blip2_image_text_matching", "coco", is_eval=True)  # 利用lavis库的 load_model 初始化blip2模型
    model.eval()
    model = model.to(device)
    return model  # 返回值是初始化（权重已经读取完毕的）Blip2


def calc_L2(vectors):
    d, measure = 256, faiss.METRIC_L2
    param = 'HNSW64'
    index = faiss.index_factory(d, param, measure)
    # print(index.is_trained)  # 此时输出为True
    index.add(vectors)
    return index


def calc_cos(vectors):
    d = 256  # 向量维度
    measure = faiss.METRIC_INNER_PRODUCT  # 使用余弦相似度
    param = 'HNSW64'
    index = faiss.index_factory(d, param, measure)  # 由于使用的是内积度量（余弦相似度），需要将索引转换为IndexFlatIP
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(vectors)  # 对向量进行L2归一化以适应余弦相似度的需求
    index.add(vectors)  # 添加向量到索引
    return index




# datasets='all'
datapath = '/mnt/disk6new/wzq/experiment/VBS/V3C1.csv'
datapath1 = '/mnt/disk6new/wzq/experiment/VBS/V3C2.csv'
datapath2 = '/mnt/disk6new/wzq/experiment/VBS/datacsv/Marine.csv'
datapath3 = '/mnt/disk6new/wzq/experiment/VBS/LHE.csv'
device = "cuda:2"  # 或者 "cpu", 根据你的设置
blip2_model = load_and_prepare_model(device)



print("Loading data ing ...")
start = time.time()
Marine_data = np.load("/mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/npdata/marine_data_norm.npy").astype(np.float32)
Marine_index = np.load("/mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/npdata/marine_index.npy")
print("读取marine完成")
print("时间：{:.3f}s".format(time.time()-start))

V3C_data = np.load("/mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/npdata/V3C_data_norm.npy").astype(np.float32)
V3C_index = np.load("/mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/npdata/V3C_index.npy")
print("读取V3C完成")
print("时间：{:.3f}s".format(time.time()-start))

# 注意： 这个LHE数据是把官方邮件里面说的 "LHE13"，"LHE19","LHE35","LHE39", "LHE40" 去掉之后的数据
LHE_data = np.load("/mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/npdata/LHE_data_norm.npy").astype(np.float32)
LHE_index = np.load("/mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/npdata/LHE_index.npy")
print("读取LHE完成")
print("时间：{:.3f}s".format(time.time()-start))
print(Marine_data.shape, V3C_data.shape, LHE_data.shape)

print("Loading time: {0:.3f} s".format(time.time()-start))


datapath4='/mnt/disk6new/wzq/experiment/VBS/VbsBackend/caption/Marine_caption.csv'
datapath5='/mnt/disk6new/wzq/experiment/VBS/VbsBackend/caption/LHE_caption.csv'
datapath6='/mnt/disk6new/wzq/experiment/VBS/VbsBackend/caption/V3C1_caption.csv'
datapath7='/mnt/disk6new/wzq/experiment/VBS/VbsBackend/caption/V3C2_caption.csv'
Marine_dict=build_combined_caption_dict(datapath4)
LHE_dict=build_combined_caption_dict(datapath5)
V3C1_dict=build_combined_caption_dict(datapath6)
V3C2_dict=build_combined_caption_dict(datapath7)

def create_index(calc_type="cos"):
    global faiss_index,faiss_index_Marine
    global faiss_index_V3C,faiss_index_LHE
    global Marine_data,V3C_data,LHE_data
    if calc_type == 'cos':
        faiss_index_Marine=calc_cos(Marine_data)
        faiss_index_V3C=calc_cos(V3C_data)
        faiss_index_LHE=calc_cos(LHE_data)
    else:
        faiss_index_Marine = calc_L2(Marine_data)
        faiss_index_V3C = calc_L2(V3C_data)
        faiss_index_LHE = calc_L2(LHE_data)

def save_index(calc_type="cos"):
    global faiss_index, faiss_index_Marine
    global faiss_index_V3C, faiss_index_LHE
    faiss.write_index(faiss_index_Marine, "/mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/index/Marine.index")
    faiss.write_index(faiss_index_V3C, "/mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/index/V3C.index")
    faiss.write_index(faiss_index_LHE, "/mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/index/LHE.index")

def read_index():
    global faiss_index, faiss_index_Marine
    global faiss_index_V3C, faiss_index_LHE
    faiss_index_Marine = faiss.read_index("/mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/index/Marine.index")
    faiss_index_LHE = faiss.read_index("/mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/index/LHE.index")
    faiss_index_V3C = faiss.read_index("/mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/index/V3C.index")
index_type = 'L2'
#create_index(index_type)
# 已经创建完成
# save_index()
start_read=time.time()
read_index()
print("读取索引时间：{:.3f}s".format(time.time() - start_read))


# 根据文件路径读取图片的方法
def load_demo_image(image_path, image_size, device):
    raw_image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    image = transform(raw_image).unsqueeze(0).to(device)
    return image

def load_array_image(raw_image, image_size, device):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    image = transform(raw_image).unsqueeze(0).to(device)
    return image
"""
   注意这里把参数model去掉了，因为我们希望把model这个变量当成全局变量存在内存里面，所以不放在函数体的局部变量里
   改用了全局变量：blip2_model，这个就是Blip2的模型，需要调用的时候就直接调用
"""


def extract_text_features(caption):
    """提取查询文本的特征向量"""
    tmp=torch.rand(1,3,364,364)
    print(tmp.shape)
    print("提特征ing")
    tmp=tmp.half()
    tmp=tmp.to(device)
    with torch.no_grad():
        sample = {"image": tmp, "text_input": caption}
        _, text_feature, _ = blip2_model(sample, match_head='itc')
        text_feature = text_feature.cpu().numpy()
        print(text_feature.shape)
        return text_feature

def extract_image_features(img):
    """提取查询图片的特征向量"""
    with torch.no_grad():
        sample = {"image": img, "text_input": 'test'}
        img_feature, _, _ = blip2_model(sample, match_head='itc')
        img_feature = torch.mean(img_feature, 1)
        img_feature = img_feature.cpu().numpy()
        return img_feature

#########################################
# 正负样本反馈
#########################################
"""
传入一个文件路径，返回Blip-2提取的图片特征
注意这里的图片特征维度应该是1*32*256的张量，我的目前是用torch.mean(image_feature,1)做一次平均
"""


def feature_extract(image_dir, model):
    image_size = 364
    caption = ' '
    with torch.no_grad():
        if os.path.isfile(image_dir):
            image = load_demo_image(image_dir, image_size=image_size, device=device)
            sample = {"image": image, "text_input": caption}
            image_feature, _, _ = model(sample, match_head='itc')
            image_feature = torch.mean(image_feature, 1)
            # print(image_feature)
            return image_feature
        else:
            print(f"Warning: {image_dir} is not a valid direction")


pos_feature = np.array([[]])  # 提取正样本的特征
neg_feature = np.array([[]])  # 提取的负样本特征


def extract_pos_feature(video_frame_dir, model):
    # pos_caption.append(get_caption_by_id(caption_dict,video_frame_dir))
    pos_cpu_feature =  np.squeeze(feature_extract(video_frame_dir, model).to('cpu').numpy(),0)
    pos_cpu_feature1=pos_cpu_feature / np.sqrt(np.sum(pos_cpu_feature**2))
    global pos_feature
    if pos_feature.size == 0:
        pos_feature = np.array([pos_cpu_feature1])
    else:
        pos_feature = np.concatenate((pos_feature, [pos_cpu_feature1]), axis=0)
    # print(pos_feature)
    # print(pos_feature)


def extract_neg_feature(video_frame_dir, model):
    neg_cpu_feature = np.squeeze(feature_extract(video_frame_dir, model).to('cpu').numpy(),0)
    neg_cpu_feature1=neg_cpu_feature / np.sqrt(np.sum(neg_cpu_feature**2))
    global neg_feature
    if neg_feature.size == 0:
        neg_feature = np.array([neg_cpu_feature1])
    else:
        neg_feature = np.concatenate((neg_feature, [neg_cpu_feature1]), axis=0)
    # print(neg_feature)


#########################################
# 排序算法部分
#########################################


def search_similar_vectors(index, query_vector, k):
    """
    对给定的查询向量，找到最相似的k个向量。
    """
    D, I = index.search(query_vector.reshape(1, -1), k)  # 搜索最相似的k个向量
    return D, I###


def changeurl(text):  # 换成绝对路径
    return '/mnt/disk6new/fsw/VBS/' + text[base_length_length:]

top_vectors=[]
glob_index_1000=None
glob_index_3000=None
glob_index_5000=None
glob_index_10000=None
def build_index_num(num):
    return calc_L2(np.array(top_vectors[:num]))
# map=[]
def thread_func_1000():
    global glob_index_1000
    glob_index_1000= build_index_num(1000)
    print("1000")
def thread_func_3000():
    global glob_index_1000
    glob_index_1000= build_index_num(3000)
    print("3000")
def thread_func_5000():
    global glob_index_1000
    glob_index_1000= build_index_num(5000)
    print("5000")
def thread_func_10000():
    global glob_index_1000
    glob_index_1000= build_index_num(10000)
    print("10000")
def norm(tmp):
    if tmp<0:
        return 0
    elif tmp>1:
        return 1
    else:
        return tmp
def find_top_similar_vectors(query_vector, k, lamda,refresh,datasets):
    """
    找到与查询向量最相似的前200个向量，并考虑正负样本的影响。
    """
    global faiss_index, faiss_index_Marine, faiss_index_V3C, faiss_index_LHE
    labels = 0
    if datasets == 'marine':
        index = faiss_index_Marine
        vectors = Marine_data
        labels = Marine_index
    elif datasets == 'V3C':
        index = faiss_index_V3C
        vectors = V3C_data
        labels = V3C_index
    elif datasets == 'LHE':
        index = faiss_index_LHE
        vectors = LHE_data
        labels = LHE_index
    else:
        index = faiss_index_V3C
        vectors = V3C_data
        labels = V3C_index

    global pos_feature,neg_feature
    starttime = datetime.datetime.now()
    print("start time: ", starttime)
    
    faiss.normalize_L2(query_vector)
    distances, indices = search_similar_vectors(index, query_vector, k)  # 搜索最相似的k个向量
    print("相似度计算返回：",distances.shape, indices.shape)
    # 计算正负样本的分数
    # 结果处理
    if refresh==0:
        global top_vectors
        global glob_index
        glob_index=None
        top_vectors = vectors[indices[0]]
        # 创建一个锁，用于同步访问glob_index
        lock = threading.Lock()
        # 创建并启动线程
        thread_1000 = threading.Thread(target=thread_func_1000)
        thread_3000 = threading.Thread(target=thread_func_3000)
        thread_5000 = threading.Thread(target=thread_func_5000)
        thread_10000 = threading.Thread(target=thread_func_10000)
        thread_1000.start()
        thread_3000.start()
        thread_5000.start()
        thread_10000.start()
    print(distances)
    distances[0] = [min(2, d) * lamda for d in distances[0]]
    
    if pos_feature.size > 1 or neg_feature.size > 1:
        glob_index=None
        if glob_index_10000!=None:
            glob_index=glob_index_10000
        elif glob_index_5000!=None:
            glob_index=glob_index_5000
        elif glob_index_3000!=None:
            glob_index=glob_index_3000
        else:
            while glob_index_1000 is None:
                time.sleep(0.001)
            glob_index=glob_index_1000
            
    if pos_feature.size>1:
        print("正反馈重算")
        add_distances, add_indices = glob_index.search(pos_feature, k)
        for pos in range(len(pos_feature)):
            tnt = 0
            for i in add_indices[pos]:
                if i == -1:
                    continue
                delta = (1 - lamda) * norm(add_distances[pos][tnt]) / len(pos_feature)
                distances[0][i] += delta
                tnt += 1
    if neg_feature.size>1:
        print("负反馈重算")
        add_distances, add_indices = glob_index.search(neg_feature, k)
        for neg in range(len(neg_feature)):
            tnt = 0
            for i in add_indices[neg]:
                if i == -1:
                    continue
                delta = (1 - lamda) * norm(add_distances[neg][tnt]) / len(neg_feature)
                distances[0][i] -= delta
                tnt += 1
    
    end = datetime.datetime.now()
    print("end time: ", end)
    combined = sorted(zip(distances[0], indices[0]))
    
    sorted_distances, sorted_indices = zip(*combined)
    top_labels = [labels[i] for i in sorted_indices]
    # 解压排序后的列表
    # 返回排序后的结果
    sorted_indices = np.argsort(sorted_distances)
   
    tot=0
    print(top_labels[0:100],"\n",sorted_distances[0:100])
    return [(top_labels[i], sorted_distances[i]) for i in sorted_indices]


inner_url = "http://172.16.15.187:8002/static/"
base_length_length = len(inner_url)
lamda = 0.5  # 超参数
result=[]

# 时间 2024-01-28 23:24:33 back_url去重方法
def backurl_unique():
    images = []
    # 唯一集合
    unique_set = set()
    for row in result:
        # 每次只会返回200个关键帧，增加返回的图片到350
        if len(images) >= 350:
            break
        row_split = row[0].split('/')[0]
        # 一个文件夹作为一个视频，即去除最后一个图片名再连接起来即可
        unique_name = "/".join(row[0].split('/')[0:-1])
        # 如果出现过在唯一集合，就跳过
        if unique_name in unique_set:
            continue
        # 否则，加到唯一集合内部
        unique_set.add(unique_name)
        if row_split == 'Marine_keyframes':
            marine = "hkust-vgd.ust.hk/Marine_frames_processed2/" + "/".join(row[0].split("/")[1:])
            images.append(inner_url + marine)
        elif row_split == 'LHE_keyframes':
            LHE = "VBSLHE_keyframes1/" + "/".join(row[0].split("/")[1:])
            images.append(inner_url + LHE)
        elif row_split == 'V3C1_keyframes':
            V3C1 = "VBSDataset/V3C1_keyframes/" + "/".join(row[0].split("/")[1:])
            images.append(inner_url + V3C1)
        else:
            V3C2 = "VBSDataset/V3C2_keyframes/keyframes/" + "/".join(row[0].split("/")[1:])
            images.append(inner_url + V3C2)  #
    return images

# 2024-01-29 13:23:20 补充修改了，因为重复图片判断，因为提特征的时候可能有一个图片的特征是重复提了好多次
def backurl():
    images = []
    for row in result:
        
        if len(images) >= 200:
            break
        
        row_split = row[0].split('/')[0]
        if row_split == 'Marine_keyframes':
            marine = "hkust-vgd.ust.hk/Marine_frames_processed2/" + "/".join(row[0].split("/")[1:])
            if inner_url + marine not in images:
                images.append(inner_url + marine)
        elif row_split == 'LHE_keyframes':
            LHE = "VBSLHE_keyframes1/" + "/".join(row[0].split("/")[1:])
            if inner_url + LHE not in images:
                images.append(inner_url + LHE)
        elif row_split == 'V3C1_keyframes':
            V3C1 = "VBSDataset/V3C1_keyframes/" + "/".join(row[0].split("/")[1:])
            if inner_url + V3C1 not in images:
                images.append(inner_url + V3C1)
        else:
            V3C2 = "VBSDataset/V3C2_keyframes/keyframes/" + "/".join(row[0].split("/")[1:])
            if inner_url + V3C2 not in images:
                images.append(inner_url + V3C2)  #
    return images
    
text_discribe=''
# 2024-01-28 23:13:20 更新, 应该需要加一个初始化吧，一开始直接在函数内部global也可以吗？
text_feature = 0
image_feature = 0 
# 2024-01-28 23:47:50 更新，补充task判断，AVS任务，一个视频最多只会返回一个关键帧
def searchListByText(text, datatype,refresh, task):
    global text_feature,pos_feature,neg_feature,text_discribe
    if refresh == 0:
        pos_feature = np.array([[]])  # 提取正样本的特征
        neg_feature = np.array([[]])  # 提取的负样本特征
        text_discribe=text
        text_feature = extract_text_features(text)
        print("特征提取结束")
    global result
    print("排序中")
    result = find_top_similar_vectors(text_feature, 10000, lamda,refresh,datatype)
    print("排序结束")
    # AVS才会要backurl_unique
    if task == "AVS":
        images = backurl_unique()
    else:
        images=backurl()
    return images

def image_preprocess(uploaded_file):
    pil_image = Image.open(uploaded_file).convert('RGB')
    return pil_image
def searchListByImage(image_array, datatype,refresh):
    global image_feature,pos_feature,neg_feature
    if refresh == 0:
        img = load_array_image(image_array, 364, device)
        pos_feature = np.array([[]])  # 提取正样本的特征
        neg_feature = np.array([[]])  # 提取的负样本特征
        image_feature = extract_image_features(img)
    print(image_feature.shape)
    global result
    result = find_top_similar_vectors(image_feature, 10000, lamda, refresh,datatype)
    images = backurl()
    return images
def searchVQA(image_array,text):
    answer_blip2= Blip2QA.blip2qa_pil(image_array,text)
    return answer_blip2

def text_to_json_list(data):
    # 查找第一个 '{' 的索引
    first_bracket_index = data.find('{')
    # 查找最后一个 '}' 的索引
    last_bracket_index = data.rfind('}')
    data = data[first_bracket_index:last_bracket_index+1]
    # 使用正则表达式提取JSON字典
    pattern = r'{[^}]*}'
    json_dicts = re.findall(pattern, data)
    scenes = []
    for scene in json_dicts:
        scenes.append(json.loads(scene))
    return scenes
pos_caption=[]
neg_caption=[]
qa_dict={}



# channel: 0 是第一个GPT4代理， channel:1 是第二个GPT代理
def send_gpt(modeltype, channel = 0):
    global questions,qa_dict
    if channel == 0:
        client = OpenAI(base_url="https://model.aigcbest.top/v1",
                    api_key="sk-GVPIhmBB4Po6mUsT8f857b6e37A84cAfAb78B848B877F7F5",
                    )
    elif channel ==1:
        client = OpenAI(base_url="http://usd.hi-cat.top/v1" ,
               api_key="sk-yc3UgOf5vk8sjN2f3b754e2c9f58430e994fDeBf86B6DfAd"
               )   
    else:
        client = OpenAI(base_url="https://model.aigcbest.top/v1",
                    api_key="sk-GVPIhmBB4Po6mUsT8f857b6e37A84cAfAb78B848B877F7F5",
                    ) 
    # Demo 1 纯文本
    text=''
    text+='The initial query is:'+text_discribe
    text+=".The positive sample text descriptions are:"
    for pos in pos_caption:
        text+=pos+','
    text+=".The negative sample text description are:"
    for neg in neg_caption:
        text+=neg+','
    text+=".You must follow standard examples given and ask three true or false questions on the central words of the above information to capture the"
    text+="user's retrieval intent in the following format:"
    text+="Answer' s format must be like this example with no other more words in triple quotes(Attention: I hope I can use json.load(line) to get this json data, so it shouldn't contains \" in value's string of json.):\n{\"id\":1,\"Q\":\"Does picture have a snake influence the description\",\"A1\":\"Yes,you need emphasize snake\",\"A2\":\"No,the problem is no snake\"}"
    print(text)
    answer=''
    
    if modeltype=='gpt':
        answer = GPT.GPT4QA(client, text, model_type='gpt-4-1106-preview')
    elif modeltype == "qwen-turbo":
        answer = Qianwen.QianwenQA(text, model_type = "qwen-turbo")
    elif modeltype == "qwen-max":
        answer = Qianwen.QianwenQA(text, model_type = "qwen-max")
    print(answer)
    tmp=text_to_json_list(answer)
    qa_dict = {}
    for item in tmp:
        qa_dict[item['id']] = {
            "Q": item['Q'],
            "Yes": item['A1'],
            "No": item['A2']
            }
    print(qa_dict)
    return qa_dict
def change_gpt(choices,modeltype, channel=0):
    print(choices)
    if channel == 0:
        client = OpenAI(base_url="https://model.aigcbest.top/v1",
                    api_key="sk-GVPIhmBB4Po6mUsT8f857b6e37A84cAfAb78B848B877F7F5",
                    )
    elif channel ==1:
        client = OpenAI(base_url="http://usd.hi-cat.top/v1" ,
               api_key="sk-yc3UgOf5vk8sjN2f3b754e2c9f58430e994fDeBf86B6DfAd"
               )   
    else:
        client = OpenAI(base_url="https://model.aigcbest.top/v1",
                    api_key="sk-GVPIhmBB4Po6mUsT8f857b6e37A84cAfAb78B848B877F7F5",
                    ) 
    cnt=0
    print(qa_dict)
    text = 'The initial description is:' + text_discribe+",consider the discription here are several question and answer"
    for id in qa_dict:
        # print(cnt)
        # print(qa_dict[id])
        # print(choices[cnt])
        text += ", Question: " + qa_dict[id]['Q'] + ". Answer: "
        if choices[cnt] == 1:
            text += qa_dict[id]['Yes']
        else:
            text += qa_dict[id]['No']
        cnt += 1
    text+=",considering these question and answer,please give a more precise discription but no more than 40 words and no more other talk."
    print(text)
    try:
        if modeltype == 'gpt':
            answer = GPT.GPT4QA(client, text, model_type='gpt-4-1106-preview')
        elif modeltype == "qwen-turbo":
            answer = Qianwen.QianwenQA(text, model_type = "qwen-turbo")
        elif modeltype == "qwen-max":
            answer = Qianwen.QianwenQA(text, model_type = "qwen-max")

    except Exception as e:
        print("Can get result from GPT") 
    print(answer)
    return answer

def solvecaption(sample,type):
    global pos_caption,neg_caption
    caption_id = sample[base_length_length:]
    idsplit = caption_id.split('/')
    flag=0
    if type == 'checkmark':
        flag=1
    if idsplit[0] == 'hkust-vgd.ust.hk' :
        idsearch='Marine_keyframes/'+'/'.join(idsplit[2:])
        if flag==1:
            tmp=get_caption_by_id(Marine_dict,idsearch)
            pos_caption.append(tmp)
        else:
            tmp=get_caption_by_id(Marine_dict, idsearch)
            neg_caption.append(tmp)
    elif idsplit[0] == 'VBSLHE_keyframes1' :
        idsearch = 'LHE_keyframes/' + '/'.join(idsplit[1:])
        print(idsearch)
        if flag == 1:
            pos_caption.append(get_caption_by_id(LHE_dict, idsearch))
        else:
            neg_caption.append(get_caption_by_id(LHE_dict, idsearch))
    elif idsplit[1] == 'V3C1_keyframes' :
        idsearch = caption_id
        print(idsearch)
        if flag == 1:
            pos_caption.append(get_caption_by_id(V3C1_dict, idsearch))
        else:
            neg_caption.append(get_caption_by_id(V3C1_dict, idsearch))
    elif idsplit[1] == 'V3C2_keyframes' :
        idsearch = caption_id
        print(idsearch)
        if flag == 1:
            pos_caption.append(get_caption_by_id(V3C2_dict, idsearch))
        else:
            neg_caption.append(get_caption_by_id(V3C2_dict, idsearch))
# tmp1 = 'http://172.16.15.187:8002/static/hkust-vgd.ust.hk/Marine_frames_processed2/Ambon_Apr2012/0003/2.png'

# solvecaption(tmp1,'checkmark')
def feedback(sample, type):
    solvecaption(sample,type)
    if type == 'checkmark':
        extract_pos_feature(changeurl(sample), blip2_model)
    elif type == 'cross':
        extract_neg_feature(changeurl(sample), blip2_model)


def extract_number(file_name):
    """ 提取文件名中的数字并转换为整数 """
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else None
    
def extract_number_V3C(file_path):
    segments = file_path.split("/")  # Split the path by '/'
    last_segment = segments[-1]  # Get the last segment
    V3C_segments = last_segment.split("_")  # Split the last segment by '_'
    numstr = V3C_segments[1]  # The number is expected to be after the underscore
    return int(numstr)  # Convert to integer and return


def findNextFolder(sample):
    sample_static = changeurl(sample)
    print(sample_static)
    choose_img = []
    if os.path.isfile(sample_static):
        # if os.path.isfile(sample_static):
        # print(sample_static)
        segments = sample_static.split("/")  # 选中图片路径分割
        numstr=0
        print(segments)
        if 'hkust-vgd.ust.hk' in sample_static or 'VBSLHE_keyframes1' in sample_static:
            numstr = re.match(r'\d+', segments[-1]).group()
        else:
            V3C_segments=segments[-1].split("_")
            numstr=V3C_segments[1]
        # print(numstr)
        num = int(numstr)
        print(num)
        nummin = max(num - 30, 0)
        result = "/".join(segments[:-1])
        last_len = len(segments[-1])
        base_seg = sample[:len(sample) - last_len]
        print(base_seg)
        image_files = [f for f in os.listdir(result)]
        # 对图片文件进行排序
        if 'hkust-vgd.ust.hk' in sample_static or 'VBSLHE_keyframes1' in sample_static:
            image_files.sort(key=extract_number)
        else:
            image_files.sort(key=extract_number_V3C)
        tot = 0
        for img_path in image_files:
            if tot < nummin:
                tot += 1
                continue
            if tot > num + 30:
                tot += 1
                continue
            tot += 1
            print(img_path)
            img_seg = img_path.split("/")
            choose_img.append(base_seg + img_seg[-1])
        return choose_img
def findAllFolder(sample):
    sample_static=changeurl(sample)
    choose_img=[]
    if os.path.isfile(sample_static):
        # print(sample_static)
        segments = sample_static.split("/") #选中图片路径分割
        # print(type(segments[-1]))
        result = "/".join(segments[:-1])
        last_len=len(segments[-1])
        base_seg=sample[:len(sample)-last_len]
        image_files = [f for f in os.listdir(result)]
        # 对图片文件进行排序
        image_files.sort(key=extract_number)
        for img_path in image_files:
            # print(img_path)
            img_seg=img_path.split("/")
            choose_img.append(base_seg+img_seg[-1])
        return choose_img



def clear():
    global pos_feature,neg_feature,pos_caption,neg_caption
    pos_feature = np.array([[]])  # 提取正样本的特征
    neg_feature = np.array([[]])  # 提取的负样本特征
    pos_caption = []
    neg_caption = []

def cut(startImage, endImage, user="user1"):
    video_id, start1, end1 = get_start_end_by_path(startImage)
    video_id, start2, end2 = get_start_end_by_path(endImage)
    print("切分视频-{0}-start from {1} ms , end in {2} ms! ".format(video_id, start1, end2))
    if split_video_for_different_user(startImage, endImage, user):
        urls = return_cut_urls_different_user(user)
        return urls

def get_videoId_time(startImage, endImage):
    video_id, start1, end1 = get_start_end_by_path(startImage)
    video_id, start2, end2 = get_start_end_by_path(endImage)
    return video_id , start1, end2
    
if __name__=='__main__':
    print()
    # text_discribe = 'a white snake is swimming in the coral reef'
    # tmp1 = 'http://172.16.15.187:8002/static/VBSDataset/V3C1_keyframes/00001/shot00001_1_RKF.png'
    # solvecaption(tmp1, 'checkmark')
    # print(pos_caption)
    # tmp2 = 'http://172.16.15.187:8002/static/VBSDataset/V3C2_keyframes/keyframes/07478/shot07478_1_RKF.png'
    # solvecaption(tmp2, 'cross')
    # print(neg_caption)
    # text_discribe='a white snake is swimming in the coral reef'
    # searchListByText(text_discribe,'marine',0)
    # tmp1='http://172.16.15.187:8002/static/hkust-vgd.ust.hk/Marine_frames_processed2/Ambon_Apr2012/0003/2.png'
    # tmp2='http://172.16.15.187:8002/static/hkust-vgd.ust.hk/Marine_frames_processed2/Ambon_Apr2012/0002/1.png'
    # tmp3='http://172.16.15.187:8002/static/hkust-vgd.ust.hk/Marine_frames_processed2/Manza_Feb2020/0032/23.png'
    # tmp4='http://172.16.15.187:8002/static/hkust-vgd.ust.hk/Marine_frames_processed2/NusaPenida_Jul2022/0053/23.png'
    # solvecaption(tmp1,'checkmark')
    # solvecaption(tmp4,'cross')
    # send_gpt()
    # tmp_choice=[1,1,1]
    # change_gpt(tmp_choice)
    # solvecaption(tmp2,'checkmark')
    # solvecaption(tmp3,'cross')
    # send_gpt()
    # tmp_choice=[1,1,1]
    # change_gpt(tmp_choice)
# tmp1='http://172.16.15.187:8002/static/hkust-vgd.ust.hk/Marine_frames_processed2/Ambon_Apr2012/0003/2.png'
# tmp2='http://172.16.15.187:8002/static/hkust-vgd.ust.hk/Marine_frames_processed2/Ambon_Apr2012/0002/1.png'
# tmp3='http://172.16.15.187:8002/static/hkust-vgd.ust.hk/Marine_frames_processed2/Manza_Feb2020/0032/23.png'
# tmp4='http://172.16.15.187:8002/static/hkust-vgd.ust.hk/Marine_frames_processed2/NusaPenida_Jul2022/0053/23.png'
# feedback(tmp1,'checkmark')
# feedback(tmp2,'checkmark')
# feedback(tmp3,'cross')
# feedback(tmp4,'cross')
# res=searchListByText('',1)
# print("check1")
# # for i in res:
# #     print(i)
# imgtmp=image_preprocess('2.png')
# print(imgtmp)
# res=searchListByImage(imgtmp,0)
# # for i in res:
# #     print(i)
# res=searchListByText('',1)
# for i in res:
#     print(i)
# feedback()