import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random
def concat_vh(img, num_img): 
    list_2d = []
    for i in range(num_img):
      list_2d.append(img)
      # return final image 
    return cv2.vconcat([cv2.hconcat(list_h)  
                        for list_h in list_2d])
def create_texture_canvas(width, height, json_path, custom_shift = None, herringbone = None):
    '''
    Функция заполняет полотно текстурой на основании ее параметров.
      width - желаемая ширина полотна в сантиметрах;
      height - желаемая высота полотна в сантиметрах;
      json_path - путь к JSON файлу с параметрами текстуры;
      custom_shift - смещение отличное от параметров текстуры по умолчанию
    '''

    def random_fragment():
        # Создаем фрагмент
        fragment = random.choice(all_fragments_images)
        rotate = random.choice(rotate_codes)
        # При необходимости переворачиваем
        if rotate:
            fragment = cv2.rotate(fragment, rotate)
        return fragment
    # Загружаем данные из json
    with open(os.path.join(json_path), 'r') as f:
        data = json.load(f)
    # Извлекаем предполагаемое имя текстуры
    name, _ = os.path.splitext(json_path)
    # Создаем пути к файлам
    if data['multiple']:
        list_of_fragments = [name + '_' + ('00'+str(i))[-2:] for i in range(data['multiple'])]
    else:
        list_of_fragments = [name]
    # Определяем расширение файла текстуры
    for ext in ['.jpg','.png']:
        path = list_of_fragments[0] + ext
        if os.path.exists(path):
            extention = ext
            break
    # Если текстура не нашлась, выдаем ошибку
    if not extention:
        print(f'Текстура {list_of_fragments[0]} не найдена')
    # Выполняем основной цикл подготовки полотна
    else:
        # Загружаем все фрагменты
        all_fragments_images = []
        for fragment_path in list_of_fragments:
          all_fragments_images.append(cv2.imread(fragment_path + extention))
        # Определяем разрешение текстуры
        texture_height, texture_width, _ = all_fragments_images[0].shape
        # Вычисляем размеры текстуры в см
        real_width = data['real_width']
        real_height = data['real_width']*texture_height/texture_width
        # Вычисляем размер полотна в пикселах
        field_width = int(width*texture_width/real_width)
        field_height = int(height*texture_height/real_height)
        # Определяем необходимоть заполнения "елочкой"
        if herringbone:
            herringbone = herringbone
        else:
            try:
                herringbone = data['herringbone']
            except:
                herringbone = None
        # Определяем возможные повороты текстуры
        rotate_codes = [None]
        # Если повороты разрешен, то добавляем повороты на 180 градусов
        if data['rotate']:
            rotate_codes.append(1)
            # Если текстура квадратная, то добавляем 90 по и против часовой стрелки
            if texture_height == texture_width and not herringbone:
                rotate_codes.append(0)
                rotate_codes.append(1)
        # Определям цикл обработки в зависимости от потребности заполнения полотна "елочкой"
        if not herringbone:
            # Вычисляем минимальное количество текстур необходимое, чтобы замостить заданную площадь
            num_x = int(width//real_width) + 1
            num_y = int(height//real_height) + 1
            # Определяем смещение текстур
            if custom_shift:
              shift = custom_shift
            else:
              shift = data['shift']
            # Инициализируем счетчик смещения при необходимости
            if shift:
              shift_counter = 0
              num_x += 1
            # Проходимся циклом по текстурам и мостим площадь
            for y in range(num_y):
                for x in range(num_x):
                    # Создаем фрагмент
                    fragment = random_fragment()
                    # Собираем полоску фрагментов
                    if 'x_line' in locals():
                        x_line = np.hstack((x_line,fragment))
                    else:
                        x_line = fragment.copy()
                # При необходимости сдвигаем
                if shift:
                    dx = int(texture_width*shift_counter*shift/100%texture_width)
                    x_line = x_line[:,texture_width-dx:texture_width*num_x-dx,:]
                    shift_counter += 1
                # Собираем горизонтальные полоски в текстуру
                if 'y_line' in locals():
                    y_line = np.vstack((y_line, x_line))
                else:
                    y_line = x_line.copy()
                del x_line
            return y_line[:field_height,:field_width,:]
        # Если нужна "елочка"
        else:
            # Вычисление основных переменных
            f_w = field_width
            f_h = field_height
            t_l = texture_width
            t_w = texture_height
            t_w = t_w*herringbone
            if t_l < t_w:
                t_w, t_l = t_l, t_w
                t_rot = True
            else:
                t_rot = False
            # Создание шаблона главной диагонали
            main_diag = np.array([[h,h] for h in range(0,f_h+t_l,t_w)])
            # Корректировка длины главной диагонали
            to_max_x = (f_w-main_diag[-1][0])//t_w
            to_max_y = (f_h-main_diag[-1][1])//t_w
            # Добавление скорректированой главной диагонали в полотно точек
            grid = main_diag.copy()[:min(to_max_x,to_max_y)]
            # Цикл для копирования диагонали в направлении нижнего края
            for h in range(1,len(main_diag)):
              # Смещения диагонали по x и y
              d_x = h*(t_l-t_w)
              d_y = h*(t_l+t_w)
              # Вылел с левого края
              #to_zero = (d_x-t_l+t_w)//t_w
              delta = (d_x-t_l+t_w)//t_w*t_w
              # Расстояние от последней планки до края полотна (y координата последней планки минус ширина полотна c запасами)
              to_max_x = (f_w-(main_diag[-1][0]-d_x+delta))//t_w
              to_max_y = (f_h-(main_diag[-1][1]+d_y+delta))//t_w
              if to_max_x < 0 or to_max_y < 0:
                  grid = np.vstack((grid,(main_diag+[-d_x+delta,d_y+delta])[:min(to_max_x,to_max_y)]))
              else:
                  grid = np.vstack((grid,(main_diag+[-d_x+delta,d_y+delta])[:]))
            # Цикл для копирования диагонали в направлении правого края
            for h in range(1,len(main_diag)):
              # Смещения диагонали по x и y
              d_x = h*t_l
              d_y = h*t_l
              # Возвратное смещение диагонали для компенсации смещения вверх
              delta = (d_y-t_l)//t_w*t_w
              # Расстояние от последней планки до края полотна (y координата последней планки минус ширина полотна c запасами)
              to_max_x = (f_w-(main_diag[-1][0]+d_x+delta))//t_w
              to_max_y = (f_h-(main_diag[-1][1]-d_y+delta))//t_w
              # Корректировка количество точек диагонали, чтобы она не выходила за правый край
              if to_max_x < 0 or to_max_y < 0:
                  grid = np.vstack((grid,(main_diag+[d_x+delta,-d_y+delta])[:min(to_max_x,to_max_y)]))
              else:
                  grid = np.vstack((grid,main_diag+[d_x+delta,-d_y+delta]))
            image = np.zeros((f_h+2*t_l+2*t_w,f_w+2*t_l,3))
            grid = grid + [t_l,t_l+t_w]
            for point in grid.tolist():
              point = point
              xh_s = point[0]
              yh_s = point[1]
              xv_s = point[0]
              yv_s = point[1]+t_w
              xh_e = xh_s+t_l
              yh_e = yh_s+t_w
              xv_e = xv_s+t_w
              yv_e = yv_s+t_l
              if t_rot:
                  img_block_h = cv2.rotate(random_fragment(),0)
                  img_block_v = random_fragment()
              else:
                  img_block_h = random_fragment()
                  img_block_v = cv2.rotate(random_fragment(),0)
              for _ in range(herringbone-1):
                    if t_rot:
                        img_block_h = np.hstack((img_block_h, cv2.rotate(random_fragment(),0)))
                        img_block_v = np.vstack((img_block_v, random_fragment()))
                    else:
                        img_block_h = np.vstack((img_block_h, random_fragment()))
                        img_block_v = np.hstack((img_block_v, cv2.rotate(random_fragment(),0)))
              # Размещение сгенерированной текстуры на полотне
              image[yh_s:yh_e,xh_s:xh_e,:] = img_block_h
              image[yv_s:yv_e,xv_s:xv_e,:] = img_block_v
            return(image[t_l+t_w:-t_l-t_w,t_l:-t_l,:])
         
def fill_texture_in_perspective(texture, original_image, surface, perspective_const = 0.45, alpha_1 = 0 ):
    if texture is None:
        print("Не удалось загрузить текстурное изображение.")
        exit()
    width, height, _ = texture.shape
    # Размеры текстуры и полотна
    texture_size = 1000
    plane_width = max(width, height)
    #if plane_width*0.35<height:
    #    plane_width=int(height/0.35)
    plane_height = plane_width
    
    # Создание пустого полотна
    #plane = np.zeros((plane_height, plane_width, 3), dtype=np.uint8)
    plane = texture
    # Наложение текстуры на полотно с учетом граничных условий
    # Наложение текстуры на полотно с учетом граничных условий
    
    #plt.imshow(plane[:,:,::-1])
    #plt.show()
    h1 = 0.4 * perspective_const
    h2 = .9 - h1
    w1 = (1 - 0.5 * perspective_const)/2-0.2
    w2 = 1 - w1
    # Координаты исходных точек
    src_points = np.float32([
        [0, 0],
        [plane_width, 0],
        [0, plane_height],
        [plane_width, plane_height]
    ])
    
    angle = 1.5 * np.pi
    if surface == 'floor':
      # Координаты целевых точек
        dst_points = np.float32([
            [plane_width * w1 + plane_width*0.3*alpha_1, plane_height - (plane_height * h1)],  # Верхний левый угол смещен вправо
            [plane_width * w2 + plane_width*0.3*alpha_1, plane_height - (plane_height * h1)],  # Верхний правый угол смещен влево
            [0, plane_height],         # Нижний левый угол без изменений
            [plane_width, plane_height]    # Нижний правый угол без изменений
        ])
    if surface == 'wall':
      # Координаты целевых точек
        dst_points = np.float32([
            [plane_width * w1, plane_height * h1],  # Верхний левый угол смещен вправо
            [plane_width * w2, plane_height * h1],  # Верхний правый угол смещен влево
            [plane_width, plane_height],    # Нижний правый угол без изменений
            [0, plane_height]         # Нижний левый угол без изменений
        ])
    if surface == 'ceiling':
        # Координаты целевых точек
        dst_points = np.float32([
            [0, 0],  # Верхний левый угол без изменений
            [plane_width, 0],  # Верхний правый угол без изменений
            [plane_width * w1, plane_height - (plane_height * h2)],    # Нижний правый угол смещен влево
            [plane_width * w2, plane_height - (plane_height * h2)]         # Нижний левый угол смещен вправо
        ])
    # Вычисление матрицы перспективного преобразования
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # Применение перспективного преобразования
    warped_image = cv2.warpPerspective(plane, M, (plane_width, plane_height))
    
    # Координаты для вырезания прямоугольника
    if surface == 'floor':
        x, y = int(plane_width * w1), int(plane_height * h1)
        width, height = int(plane_width * w2 - plane_width * w1), int(plane_height - plane_height * h1)
    else:
        x, y = int(plane_width * w1), 0
        width, height = int(plane_width * w2 - plane_width * w1), int(plane_height)
    # Вырезание прямоугольника из изображения
    if surface == 'wall':
        texture_image = cv2.getRectSubPix(plane, (width, height), (x + width/2, y + height/2))
    elif surface == 'floor':
        texture_image = cv2.getRectSubPix(warped_image, (width, height), (x + width/2, y + height/2))
    elif surface == 'ceiling':
        texture_image = cv2.getRectSubPix(warped_image, (width, height), (x + width/2, y + height/2))
    print('texture_image.shape = ',texture_image.shape)
    return cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB),cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

def transform_texture(original_image, texture_image, floor_mask, surface = 'floor', overlay= False, alpha = 0.2):
    # print('original_image.shape,floor_mask.shape, texture_image.shape',original_image.shape,floor_mask.shape, texture_image.shape)

    original_image = np.array(original_image)
    
    # Пропорциональное масштабирование текстуры под исходное изображение
    scale_factor = max(original_image.shape[0] / texture_image.shape[0], original_image.shape[1] / texture_image.shape[1])*1.5

    #rescaled_texture = cv2.resize(texture_image, (int(texture_image.shape[1] * scale_factor), int(texture_image.shape[0] * scale_factor)))
    rescaled_texture = texture_image
    # Нахождение центров масштабированной текстуры и исходного изображения
    center_original = (original_image.shape[1] // 2, original_image.shape[0] // 2)
    center_texture = (rescaled_texture.shape[1] // 2, rescaled_texture.shape[0] // 2)

    # Определение области обрезки для текстуры

    start_x = max(center_texture[0] - center_original[0], 0)
    start_y = max(center_texture[1] - center_original[1], 0)
    end_x = min(center_texture[0] + original_image.shape[1] - center_original[0], rescaled_texture.shape[1])
    end_y = min(center_texture[1] + original_image.shape[0] - center_original[1], rescaled_texture.shape[0])
    
   
    print('surface = ',surface)
    # Обрезка текстуры
    texture_image = rescaled_texture[start_y:end_y, start_x:end_x]
    if surface == 'floor':
        #texture_image = rescaled_texture[rescaled_texture.shape[0]-original_image.shape[0]:rescaled_texture.shape[0], start_x:end_x]
        texture_image = rescaled_texture[rescaled_texture.shape[0]-original_image.shape[0]:rescaled_texture.shape[0], start_x:end_x]
    elif surface == 'ceiling':
        texture_image = rescaled_texture[original_image.shape[0], start_x:end_x]
    elif surface == 'wall':
        texture_image = rescaled_texture[start_y:end_y, 0:original_image.shape[1]]

    
    #texture_image = rescaled_texture[rescaled_texture.shape[0]-original_image.shape[0]:rescaled_texture.shape[0], start_x:end_x]
    # Преобразование цветных пикселей пола в серую гамму по маске пола
    gray_floor = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_floor_colored = cv2.cvtColor(gray_floor, cv2.COLOR_GRAY2BGR)  # Преобразование обратно в трехканальное изображение
    alpha = 0.2  # Пример пропорции (50% оригинальное изображение, 50% текстура)
    blended_image = original_image.copy()
    # Создание копии оригинального изображения для смешивания
    
    
    if overlay:
        blended_image_1 = cv2.addWeighted(gray_floor_colored, alpha, texture_image, 1 - alpha, 0) 
        blended_image[floor_mask > 0] = blended_image_1[floor_mask > 0]
    else:
        blended_image[floor_mask > 0] = texture_image[floor_mask > 0]
    
    # Задание пропорции смешивания
    

    # Смешивание серого пола с текстурой по маске пола
    #print('1',gray_floor_colored[floor_mask > 0].shape, texture_image[floor_mask > 0].shape)
    #blended_image[floor_mask > 0] = cv2.addWeighted(gray_floor_colored[floor_mask > 0], alpha, texture_image[floor_mask > 0], 1 - alpha, 0)
    
    return blended_image
def roted_texture(texture, angel):

    M = cv2.getRotationMatrix2D((texture.shape[1] // 2, texture.shape[0] // 2), angel, 1)
    rot = cv2.warpAffine(texture, M, (texture.shape[1], texture.shape[0]))
    h = rot.shape[0]//1.4
    w = rot.shape[1]//1.4
    center_texture = (rot.shape[1] // 2, rot.shape[0] // 2)
    start_x = int(center_texture[0] - w//2)
    start_y = int(center_texture[1] - h//2)
    end_x = int(center_texture[0] + w//2)
    end_y = int(center_texture[1] + h//2)
    #print(texture.shape, rot.shape)
    #print('start_x = ', start_x, 'start_y = ', start_y, 'end_x = ', end_x, 'end_y = ', end_y)
    texture = rot[start_y:end_y, start_x:end_x]
    
    return texture
def brith_mask(image, texture, seg_mask, blender_image):
    seg_mask = seg_mask[:,:,0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[np.where(seg_mask==0)] = 0
    gray_tex = cv2.cvtColor(texture, cv2.COLOR_RGB2GRAY)
    Gaussian = cv2.GaussianBlur(gray, (101, 101), 0)
    hist = cv2.calcHist([Gaussian], [0], None,  [256], [10, 256])
    Gaussian_tex = cv2.GaussianBlur(gray_tex, (51, 51), 0) 
    hist_tex = cv2.calcHist([Gaussian_tex], [0], None,  [256], [10, 256])
    if np.argmax(hist_tex) < np.argmax(hist):
        const = np.argmax(hist)  - (np.argmax(hist) - np.argmax(hist_tex)) +10
    elif np.argmax(hist) < np.argmax(hist_tex):
        const = np.argmax(hist) +10
    shadow = cv2.GaussianBlur(gray, (15, 25), 0) 
    brithness_mask = (shadow[np.where(seg_mask>0)].astype(np.float16) - const) * 3
    brithness_mask[np.where(brithness_mask>20)] = 20
    color_map = (blender_image[np.where(seg_mask>0)]).astype(np.float16)
    b = (color_map[:,0]/(color_map[:,0] + color_map[:,1] + color_map[:,2])).astype(np.float16)
    g = (color_map[:,1]/(color_map[:,0] + color_map[:,1] + color_map[:,2])).astype(np.float16)
    r = (color_map[:,2]/(color_map[:,0] + color_map[:,1] + color_map[:,2])).astype(np.float16)
    color_map[:,0] = (color_map[:,0] + brithness_mask * b).astype(np.float16)
    color_map[:,1] = (color_map[:,1] + brithness_mask  * g).astype(np.float16)
    color_map[:,2] = (color_map[:,2] + brithness_mask * r).astype(np.float16)
    color_map[np.where(color_map<0)] = 0
    color_map[np.where(color_map>255)] = 255
    blender_image[np.where(seg_mask>0)] = color_map.astype(np.uint8)
    return blender_image
def texture_perspective(image, texture, mask, originan_image, perspective_const = 0.35, angel = 0, alpha_1 = 0, surface = 'floor', overlay = False):
    print(perspective_const)
    texture = roted_texture(texture, angel)
    texture_image,_= fill_texture_in_perspective(texture ,image, surface, perspective_const, alpha_1)
    blended_image  = transform_texture(image,texture_image, mask, surface, overlay)
    blended_image_2 = brith_mask(originan_image, texture, mask, blended_image)
    return blended_image_2
