import torch
import mmcv
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor
from mmcv import Config
import cv2
import matplotlib.pyplot as plt
from moduls.settings import model_weigths
def edit_model():
# Скачиваем и распаковываем репозиторий, если он еще не скачан
    # Получаем список конфигурационных файлов
    config_path, checkpoint_path = model_weigths()
    # Загружаем модель один раз
    cfg = Config.fromfile(config_path)
    cfg.model.test_cfg.avg_non_ignore = True
    if 'evaluation' not in cfg:
        cfg.evaluation = dict(metric=['mIoU', 'mDice', 'mAcc', 'precision', 'recall'])
    model = init_segmentor(cfg, checkpoint_path, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    return model
def resize_image(img_path):
    print('img_path', type(img_path))
    img = cv2.imread(img_path)
    # Определить новый размер
    new_size = (1080, 1920)
# Определить пропорциональное изменение
    h, w = img.shape[:2]
    h_ratio = new_size[0] / h
    w_ratio = new_size[1] / w
    if h_ratio == 1 or w_ratio == 1:
        new_img = img
    else:
# Выбрать минимальное соотношение для масштабирования
        ratio = min(h_ratio, w_ratio)
    # Новый размер после пропорционального изменения
        new_h = int(h * ratio)
        new_w = int(w * ratio)
    # Определить смещение для центрирования
        top = (new_size[0] - new_h) // 2
        left = (new_size[1] - new_w) // 2
    # Изменить размер изображения с сохранением пропорций
        resized_img = cv2.resize(img, (new_w, new_h))
    # Создать новое черное изображение
        new_img = np.zeros(new_size + (3,), np.uint8)
        new_img[top:top + new_h, left:left + new_w] = resized_img
    
    return resized_img
def predict_and_visualize(model, image_path, dataset_name = 'ade20k'):
    # Предсказание
    color_mapping = {
        'ade20k': {0: (100), 3: (150), 5: (200)}
    }
    color_mapping_2 = {
        'ade20k':{
        3: (255, 0, 0), 28: (255, 0, 0),
        #ceiling
        5: (0, 255, 0),
        #wall
        0: (0, 0, 255), 14:(0, 0, 255), 8: (0, 0, 255), 18: (0, 0, 255), 22: (0, 0, 255), 24: (0, 0, 255), 27: (0, 0, 255), 31: (0, 0, 255), 33: (0, 0, 255), 35: (0, 0, 255), 41: (0, 0, 255), 42: (0, 0, 255),
        49: (0, 0, 255), 62: (0, 0, 255), 74: (0, 0, 255),  89: (0, 0, 255), 148: (0, 0, 255),
        #other
        7: (0, 0, 0), 10: (0, 0, 0), 15: (0, 0, 0), 17: (0, 0, 0), 19: (0, 0, 0), 23: (0, 0, 0), 35: (0, 0, 0),36: (0, 0, 0), 39: (0, 0, 0), 64: (0, 0, 0), 66: (0, 0, 0), 67: (0, 0, 0), 75: (0, 0, 0), 82: (0, 0, 0), 85: (0, 0, 0), 97: (0, 0, 0),
        112: (0, 0, 0), 125: (0, 0, 0),132: (0, 0, 0), 134: (0, 0, 0), 135: (0, 0, 0), 137: (0, 0, 0), 142: (0, 0, 0)}
    }
    #print(image_path)
    #img_r = resize_image(image_path)
    img_r = cv2.imread(image_path)
    image = mmcv.imread(img_r)
    
    result = inference_segmentor(model, image)

    # Преобразование тензоров в numpy массивы
    seg_map = result[0].cpu().numpy() if isinstance(result[0], torch.Tensor) else result[0]
    seg_floor = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    seg_floor[seg_map == 3, :] = 150
    seg_ceiling = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    seg_ceiling[seg_map == 5, :] = 200
    seg_wall  = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    seg_wall[seg_map == 0, :] = 100
    seg_all = [seg_floor, seg_wall, seg_ceiling]
    color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    gray_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for label, color in color_mapping_2[dataset_name].items():
        color_seg[seg_map == label, :] = color
    for label, color in color_mapping[dataset_name].items():
        gray_seg[seg_map == label, :] = color
    return seg_map, color_seg, seg_all, image
def predict_and_visualize_2(model, image_path, dataset_name = 'ade20k'):
    # Предсказание
    color_mapping = {
        'ade20k': {0: (100), 3: (150), 5: (200)}
    }
    data_classes = {
        'ade20k': {
    # class, id_class, class_type, color
    3: (255, 0, 0),  # Светлый розовый
    5: (0, 255, 0),  # Светлый зеленый
    0: (0, 0, 255),      # Светлый синий
    14: (139, 69, 19),        # Коричневый
    7: (255, 228, 196),        # Персиковый
    8: (173, 216, 230),     # Голубой
    10: (160, 82, 45),     # Коричневый
    15: (222, 184, 135),     # Слоновая кость
    18: (255, 182, 193),   # Розовый
    19: (240, 230, 140),     # Светло-желтый
    22: (255, 218, 185),   # Персиковый
    23: (255, 160, 122),      # Светло-коралловый
    24: (139, 137, 137),     # Темно-серый
    27: (192, 192, 192),     # Серебристый
    28: (255, 99, 71),         # Красный
    31: (100, 150, 200),      # Серый синий
    33: (255, 165, 0),        # Оранжевый
    35: (244, 164, 96),       # Светлый коричневый
    42: (255, 228, 196),    # Персиковый
    17: (100, 200, 100),     # Зеленый
    39: (255, 105, 180),   # Ярко-розовый
    67: (65, 80, 0),          # Темно-зеленый
    74: (0, 80, 65),      # Темно-бирюзовый
    75: (180, 200, 160),    # Светлый зеленый
    82: (10, 50, 110),       # Темно-синий
    89: (165, 180, 30), # Светло-оливковый
    132: (180, 135, 150),# Розовато-серый
    36: (33, 96, 186),        # Темно-синий
    62: (43, 243, 128),   # Ярко-зеленый
    64: (87, 54, 231),      # Фиолетовый
    85: (164, 87, 242), # Светло-фиолетовый
    97: (112, 222, 76),    # Ярко-зеленый
    148: (12, 22, 69),       # Темно-синий
    135: (200, 160, 140),     # Бежевый
    134: (164, 96, 244),    # Фиолетовый
    137: (30, 70, 130),        # ?
    112: (96, 164, 244),    # Светло-голубой
    142: (164, 244, 96),     # Светло-зеленый
    66: (64, 144, 6),       # Ярко-зеленый
    125: (80, 0, 65),          # Темно-бордовый
    49: (2, 22, 222),    # Темно-синий
    35: (1, 111, 11),     # Темно-зеленый
    41: (76, 143, 109),         # Темно-серый
}}
    color_mapping_2 = {
            'ade20k':{
            3: (255, 0, 0), 28: (255, 0, 0),
            #ceiling
            5: (0, 255, 0),
            #wall
            0: (0, 0, 255), 14:(0, 0, 255), 8: (0, 0, 255), 18: (0, 0, 255), 22: (0, 0, 255), 24: (0, 0, 255), 27: (0, 0, 255), 31: (0, 0, 255), 33: (0, 0, 255), 35: (0, 0, 255), 41: (0, 0, 255), 42: (0, 0, 255),
            49: (0, 0, 255), 62: (0, 0, 255), 74: (0, 0, 255),  89: (0, 0, 255), 148: (0, 0, 255),
            #other
            7: (0, 0, 0), 10: (0, 0, 0), 15: (0, 0, 0), 17: (0, 0, 0), 19: (0, 0, 0), 23: (0, 0, 0), 35: (0, 0, 0),36: (0, 0, 0), 39: (0, 0, 0), 64: (0, 0, 0), 66: (0, 0, 0), 67: (0, 0, 0), 75: (0, 0, 0), 82: (0, 0, 0), 85: (0, 0, 0), 97: (0, 0, 0),
            112: (0, 0, 0), 125: (0, 0, 0),132: (0, 0, 0), 134: (0, 0, 0), 135: (0, 0, 0), 137: (0, 0, 0), 142: (0, 0, 0)}
        }
    img_r = resize_image(image_path)
    #img_r = cv2.imread(image_path)
    image = mmcv.imread(img_r)
    result = inference_segmentor(model, image)
    # Преобразование тензоров в numpy массивы
    seg_map = result[0].cpu().numpy() if isinstance(result[0], torch.Tensor) else result[0]
    seg_floor = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    seg_floor[seg_map == 3, :] = 150
    seg_ceiling = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    seg_ceiling[seg_map == 5, :] = 200
    seg_wall  = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    seg_wall[seg_map == 0, :] = 100
    seg_all = [seg_floor, seg_wall, seg_ceiling]
    color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    color_seg_3 = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for label, color in data_classes[dataset_name].items():
        color_seg[seg_map == label, :] = color
    for label, color in color_mapping_2[dataset_name].items():
        color_seg_3[seg_map == label, :] = color
    return seg_map, color_seg, color_seg_3, seg_all, image
