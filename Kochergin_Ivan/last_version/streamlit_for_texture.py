import streamlit as st
import cv2
from streamlit_image_annotation import detection
import os
import json
from zipfile import ZipFile
from PIL import Image
import numpy as np
import re
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")
def save_file_st(path_file):
    file_name = path_file.name
    img = Image.open(path_file)
    im = np.array(img)
    output_filename = file_name
    cv2.imwrite(output_filename, cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) 
    #im.save(output_filename)
    return output_filename
def zip_files(file_list):
    print(1)
    with ZipFile("new_texture.zip", "w") as zipObj:
        for idx in file_list:
            zipObj.write(idx)
    return 'new_texture.zip'
def json_texture(bbox, path, real_width, rotate=True, shift=50):
    name = path
    image = cv2.imread(path)
    image_1 = image.copy()
    a = 0
    h_list = []
    w_list = []
    for i in bbox:
        if i['bbox'][3]> i['bbox'][2]:
            h_list.append(i['bbox'][3])
            w_list.append(i['bbox'][2])
    h = min(h_list)
    h_1 = min(w_list)
    w = min(w_list)
    w_1 = min(h_list)
    file_list = []
    for i in bbox:
        if a < 10:
            name_1 = re.split('.jpg', name)[0]
            name_t = name_1 + '_0' + str(a) + '.jpg'
            image_t2 = image.copy()
            if i['bbox'][3]> i['bbox'][2]:
                image_t = image_t2[int(i['bbox'][1]):int(i['bbox'][1]+h), int(i['bbox'][0]):int(i['bbox'][0]+w)]
            else:
                image_t = image_t2[int(i['bbox'][1]):int(i['bbox'][1]+h_1), int(i['bbox'][0]):int(i['bbox'][0]+w_1)]
                image_t = cv2.rotate(image_t, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(str(name_t), image_t)
            file_list.append(name_t)
            a += 1
        else:
            name_1 = re.split('.jpg', name)[0]
            name_t = name_1 + '_' + str(a) + '.jpg'
            if i['bbox'][3]> i['bbox'][2]:
                image_t = image_t2[int(i['bbox'][1]):int(i['bbox'][1]+h), int(i['bbox'][0]):int(i['bbox'][0]+w)]
            else:
                image_t = image_t2[int(i['bbox'][1]):int(i['bbox'][1]+h_1), int(i['bbox'][0]):int(i['bbox'][0]+w_1)]
                image_t = cv2.rotate(image_t, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(str(name_t), image_t)
            file_list.append(name_t)
            a += 1
    if image_1.shape[0] > image_1.shape[1]:
        proc = image_1.shape[0]/image_1.shape[1]
        h_pic = 400
        w_pic = int(400*proc)
        image_1 = cv2.resize(image_1, (h_pic, w_pic))
    elif image_1.shape[0] < image_1.shape[1]:
        proc = image_1.shape[1]/image_1.shape[0]
        w_pic = 400
        h_pic = int(400*proc)
        image_1 = cv2.resize(image_1, (h_pic, w_pic))
    else:
        image_1 = cv2.resize(image_1, (400, 400))
    cv2.imwrite(name_1+'_icon.jpg', image_1[:400,:400])
    file_list.append(name_1+'_icon.jpg')
    texture_parameters = {'name': name,
                        'real_width': real_width,
                            'multiple': a,
                            'rotate': rotate,
                            'shift': shift}    
    # Сохраним словать в json
    file_list.append(name_1 + '.json')
    st.session_state.file_list = file_list
    with open(name_1 +'.json', 'w') as f:
        json.dump(texture_parameters, f)
    
   
path_obj = st.file_uploader("Загрузка текстуры", type=['png','jpeg','jpg'], accept_multiple_files=False, key=None,)
if path_obj is not None:
    st.write(path_obj.name)
    path = save_file_st(path_obj)


if path_obj is not None:
    bbox = detection(path, label_list = ['Таблетка'], bboxes = [], labels=[],height = 1080, width = 1920, use_space=True)
col_1, col_2 = st.columns([1,4])
with col_1:

    st.session_state.shift = st.number_input('Перекрытие при укладке в процентах (%)', min_value=0, max_value=100, value=50, step=1)
    st.session_state.real_width = st.number_input('Реальная ширина в сантиметрах', min_value=0, max_value=1000, value=50, step=1)
    st.session_state.rotate = st.checkbox('Поворот текстуры', value=False)
if path_obj is not None:
    if bbox is None:
        st.session_state.bbox = bbox
    if 'bbox' in st.session_state and st.session_state.bbox != bbox:
        st.session_state.bbox = bbox
        json_texture(st.session_state.bbox, path, st.session_state.real_width, rotate=st.session_state.rotate, shift=st.session_state.shift)
    #if 'bbox' in st.session_state:
        #st.write(st.session_state.bbox)
with col_2:    
#btn = st.button('Скачать текстуру')
    if 'file_list' in st.session_state and st.session_state.file_list != None:    
        with ZipFile("new_texture.zip", "w") as zipObj:
            for idx in st.session_state.file_list:
                zipObj.write(idx)
        with open('new_texture.zip', "rb") as file:
                btn = st.download_button(
                            label="Скачать изображение с наложенной текстурой",
                            data=file,
                            file_name="new_texture.zip",
                            mime="application/zip"
                            )
