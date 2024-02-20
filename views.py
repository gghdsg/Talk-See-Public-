from django.shortcuts import render
import os
# Create your views here.
from django.http import JsonResponse
from django.template.response import TemplateResponse
from django.shortcuts import HttpResponse, render, redirect
import cv2
import numpy as np
from PIL import Image
import base64
import csv
from . import engine
import jpype
from .uploadResult import submit_demo

jar_dir = "/mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/uploadResult"
libjar = ":".join([jar_dir+"/lib/"+jar for jar in os.listdir(jar_dir+"/lib")])
# 启动JVM并指定JAR包路径
if not jpype.isJVMStarted():
    startJVM(getDefaultJVMPath(), "-ea", f"-Djava.class.path={jar_dir}/openapi-java-client-2.0.0-RC4.jar:"+libjar)
else:
    jpype.addClassPath(f"/mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/uploadResult/openapi-java-client-2.0.0-RC4.jar")
    jpype.addClassPath(r"/mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/uploadResult/*")


username = "user1"
eval_type = "now"


inner_url = "http://172.16.15.187:8002/static/"
tempoary_images = []
temporay_text = ""
temporay_task = ""
def home(request):
    if request.method == 'GET':
        return render(request, 'home_demo1.html', {'message': 'Welcome to the homepage'})

# Demo
def search(request):
    global tempoary_images, temporay_task
    if request.is_ajax():
#        action = request.POST.get('action')  # 用户选择的动作，rerank或search
        text = request.POST.get('text')  # 用户输入的检索文本
        task = request.POST.get('task')  # 取值为三个任务 "AVS", "KIS", "VQA"
        image=request.FILES.get('file')
        data=request.POST.get('data')
        print(text,task,data)
        images_url = []
        temporay_task = task
        
        if image and text and task=='VQA':
            pil_image = image_preprocess(image)
            print(pil_image)
            print(text)
            answer=engine.searchVQA(pil_image,text)
            json={"ans_blip2":answer}
            return JsonResponse(json)
        if image:
            pil_image = image_preprocess(image)  # 用户提交的图片，现在已经以opencv读取的numpy.array数组形式返回
            # image_array = np.array(pil_image)
            images = engine.searchListByImage(pil_image,data,0)
            tempoary_images = images
            json = {'images':tempoary_images}
            return JsonResponse(json)
        if text:
            temporay_text = text
            images = engine.searchListByText(text,data,0, temporay_task)
            tempoary_images = images
            json = {'images':tempoary_images}
            return JsonResponse(json)
        # VQA 任务

        # for path in os.listdir("/mnt/disk6new/fsw/VBS/VBSDataset/V3C2_keyframes/keyframes/07479"):
        #     images_url.append(inner_url+'VBSDataset/V3C2_keyframes/keyframes/07479/'+path)
        # json = {'images': images_url}
        json = {'images':tempoary_images}
        return JsonResponse(json)
def rerank(request):
    global tempoary_images
    if request.is_ajax():
        text = request.POST.get('text')  # 用户输入的检索文本
        task = request.POST.get('task')  # 取值为三个任务 "AVS", "KIS", "VQA"
        data= request.POST.get('data')
        print(text, task,data)
        images_url = []
        # 2024-01-28-23:51 重新修改，由于更新了AVS任务一个视频只会返回一个关键帧，这里需要分开text是否为空的判断和AVS任务
        if task == 'AVS':
            images = engine.searchListByText('',data, 1, task)
            tempoary_images = images
        if task == 'KIS':
            if text:
                images = engine.searchListByText('',data, 1, task)
                tempoary_images = images
            else:
                simple_image = Image.new('RGB', (64, 64), color=(255, 0, 0))
                images = engine.searchListByImage(simple_image,data, 1)
                tempoary_images = images
        # for path in os.listdir("/mnt/disk6new/fsw/VBS/VBSDataset/V3C2_keyframes/keyframes/07479"):
        #     images_url.append(inner_url+'VBSDataset/V3C2_keyframes/keyframes/07479/'+path)
        # json = {'images': images_url}
        json = {'images': tempoary_images}
        return JsonResponse(json)
def feedback(request):
    if request.is_ajax():
        check_file = request.POST.get('selectedImagePath')  #
        icon_type=request.POST.get('iconType')
        if temporay_task == "AVS" and icon_type == "checkmark":
            start_end = check_file
            vid, start_time, end_time = engine.get_videoId_time(start_end, start_end)
            submit_demo.submit_clip_different_user(user=username,subinfoArray=[[vid, start_time, end_time]], etype=eval_type)
        print(f"{check_file}---{icon_type}")
        engine.feedback(check_file,icon_type)
        # print(image_id)
        return JsonResponse({})

def feedTime(request):
    if request.is_ajax():
        check_file = request.POST.get('selectedImagePath')  #
        icon_type=request.POST.get('iconType')
        print(f"{check_file}---{icon_type}")
        engine.feedTime(check_file,icon_type)
        # print(image_id)
        return JsonResponse({})
def clear(request):
    if request.is_ajax():
        print('clear')
        engine.clear();
        return JsonResponse({})

def np_array_to_base64(image):
    img_array = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #RGB2BGR，用于cv2编码
    encode_image = cv2.imencode(".jpg", img_array)[1] #用cv2压缩/编码，转为一维数组
    byte_data = encode_image.tobytes() #转换为二进制
    base64_str = base64.b64encode(byte_data).decode("ascii") #转换为base64
    return base64_str
    
def displaynext(request):
    if request.is_ajax():
        img_path=request.POST.get('selectedImagePath')
        print(img_path) 
        # engine代码有误，重写
        next_images = engine.findNextFolder(img_path)
        json = {'nextImages': next_images}
        return JsonResponse(json)
        # 测试的图片传输功能
#        next_images = []
#        for path in os.listdir("/mnt/disk6new/fsw/VBS/VBSDataset/V3C2_keyframes/keyframes/07479"):
#            next_images.append(inner_url+'VBSDataset/V3C2_keyframes/keyframes/07479/'+path)
#        json = {'next_images': next_images}
#        return JsonResponse(json)

def displayall(request):
    if request.is_ajax():
        img_path=request.POST.get('selectedImagePath')
        all_images = engine.findAllFolder(img_path)
        json = {'allImages': all_images}
        return JsonResponse(json)
         # 测试的图片传输功能
#        all_images = []
#        for path in os.listdir("/mnt/disk6new/fsw/VBS/VBSDataset/V3C2_keyframes/keyframes/07479"):
#            all_images.append(inner_url+'VBSDataset/V3C2_keyframes/keyframes/07479/'+path)
#        json = {'all_images': all_images}
#        return JsonResponse(json)

def cut(request):
    if request.is_ajax():
        print("切视频请求")
        start = request.POST.get('startTimePath')
        end = request.POST.get('endTimePath') 
        print(start, end)
        urls = engine.cut(start, end, user=username)
        print(urls)
        json = {'allImages': urls}
        return JsonResponse(json)

def finish(request):
    if request.is_ajax():
        start = request.POST.get('startImagePath')
        end = request.POST.get('endImagePath') 
        vid = start.split("/")[-1].split("_")[0]
        split_start_time = int(start.split("/")[-1].split("_")[1])
        start_id = start.split("/")[-1].split("_")[2]
        end_id =  end.split("/")[-1].split("_")[2]
        print(start_id, end_id)
        start_time = int(0.5 * (int(start_id[:-4])-1) *1000) + split_start_time 
        end_time = int(0.5 * (int(end_id[:-4])-1) *1000) + split_start_time 

        print(vid, start_time, end_time)
        # 提交答案的代码， 第一个参数是提交的用户名， 第二个参数是提交的内容，第三个是提交的类型，etype是test时是测试, 正常是实时比赛，etype为now，取activate的第一个id
        # 其中evaluationid需要自己指定:  /mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/uploadResult/real_time.txt
        submit_demo.submit_clip_different_user(user= username,subinfoArray=[[vid, start_time, end_time]], etype=eval_type)
        json = {'null': 'null'}
        return JsonResponse(json)

def send(request):
    if request.is_ajax():
        start = request.POST.get('startImagePath')
        end = request.POST.get('endImagePath') 
        vid, start_time, end_time = engine.get_videoId_time(start, end)
        print(vid, start_time, end_time)
        # 提交答案的代码， 第一个参数是提交的用户名， 第二个参数是提交的内容，第三个是提交的类型，etype是test时是测试, 正常是实时比赛，etype为now，取activate的第一个id
        # 其中evaluationid需要自己指定:  /mnt/disk6new/wzq/experiment/VBS/VbsBackend/competition/uploadResult/real_time_QA.txt
        submit_demo.submit_clip_different_user(user=username,subinfoArray=[[vid, start_time, end_time]], etype=eval_type)
        
        json = {'null': 'null'}
        return JsonResponse(json)

def submitconfirm(request):
    if request.is_ajax():
        answer = request.POST.get('userInput')
        print(f"upload result: {answer}")
        submit_demo.submit_QA_different_user(user=username,answer_text=answer, etype=eval_type)
        json = {'null': 'null'}
        return JsonResponse(json)
def image_preprocess(uploaded_file):
    # 检查文件格式
    allowed_formats = ['png', 'jpg', 'jpeg']
    file_format = uploaded_file.name.split('.')[-1].lower()
    if file_format not in allowed_formats:
        return JsonResponse({'status': 'error', 'message': 'Invalid file format'})
    # 使用 Pillow 打开上传的图像
    pil_image = Image.open(uploaded_file).convert('RGB')
    return pil_image

def GPT_QA(request):
    if request.is_ajax():
        model=request.POST.get('model')
        print(model)
        answer={}
        if model=='gpt_0':
            return JsonResponse(engine.send_gpt('gpt',0))
        elif model=='gpt_1':
            return JsonResponse(engine.send_gpt('gpt',1))
        else:
            return JsonResponse(engine.send_gpt(model))


def GPT_describe(request):
    if request.is_ajax():
        check_queue= request.POST.get('answers')
        model = request.POST.get('model')
        print(model)
        answer = {}
        if model == 'gpt_0':
            return JsonResponse({'text':engine.change_gpt(check_queue,'gpt',0)})
        elif model == 'gpt_1':
            return JsonResponse({'text':engine.change_gpt(check_queue,'gpt',1)})
        else:
            return JsonResponse({'text':engine.change_gpt(check_queue,model)})
