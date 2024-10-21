import os
import cv2
import numpy as np
import random
import json
def create_texture_canvas(width, height, json_path, custom_shift = None, herringbone = None, rescale = None):
    '''
    Функция заполняет полотно текстурой на основании ее параметров указанных в JSON файле.
      width - желаемая ширина полотна (см);
      height - желаемая высота полотна (см);
      json_path - путь к JSON файлу с параметрами текстуры;
      custom_shift - смещение в направлении длины отличное от параметров текстуры по умолчанию (%);
      herringbone - укладка ёлочкой:
                        None или 0 - нет укладки ёлочкой;
                        1 и выше - ёлочка с соответствующим количеством планок;
                        Смещение при укладке ёлочкой не работает;
      recsale - коэффициент масштабирования фрагментов текстуры для экономии памяти.
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
        if not rescale:
            for fragment_path in list_of_fragments:
                all_fragments_images.append(cv2.imread(fragment_path + extention))
        else:
            texture_height, texture_width, _=  cv2.imread(list_of_fragments[0] + extention).shape
            for fragment_path in list_of_fragments:
                fragment = cv2.imread(fragment_path + extention)
                all_fragments_images.append(cv2.resize(fragment, (int(texture_width/rescale), int(texture_height/rescale))))

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
# Функция наложения текстуры по отдельному цвету маски
def build_mask(ceiling_points,
               floor_points,
               size = (800, 600),
               colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)],
               walls_color_delta = (0, 75, 0)):
    '''
    Функция построения маски по опорным точкам потолка и пола. Дополняет маску до краев изображения.
    Стены строятся по последовательным  парам точек пола и потолка.
    Работает от 3-х точек и более. Если один из списков длиннее, то оставшиеся точки игнорируются.
    ceiling_points, floor_points - списки точек пола и потолка;
    size - размер изображения маски;
    colors - список цветов потолка, пола, стен для маски;
    walls_color_delta - смещение цветовой палитры для цвета стен.
    '''

    # Извлечение размеров
    width, height = size
    # Создание полигона потолка с добавлением углов изображения
    ceiling_mask = ceiling_points.copy()
    if ceiling_points[0][0] == 0 and ceiling_points[0][1] != 0:
        ceiling_mask = [[0,0]] + ceiling_mask
    if ceiling_points[-1][0] == width and ceiling_points[-1][1] != 0:
        ceiling_mask = ceiling_mask + [[width,0]]

    # Создание полигона пола с добавлением углов изображения
    floor_mask = floor_points.copy()
    if floor_points[0][0] == 0 and floor_points[0][1] != height:
        floor_mask = [[0,height]] + floor_mask
    if floor_points[-1][0] == width and floor_points[-1][1] != height:
        floor_mask = floor_mask + [[width,height]]

    # Создание полигонов стен с добавлением углов изображения
    wall_masks = []
    for i in range(0,min(len(ceiling_points),len(floor_points))-1):
        wall = [ceiling_points[i],floor_points[i],floor_points[i+1],ceiling_points[i+1]]
        wall_masks.append(wall)

    # Корекция точек маски левой стены
    if wall_masks[0][1][1] == height and wall_masks[0][1][0] != 0:
        wall_masks[0] = wall_masks[0][:1] + [[0,height]] + wall_masks[0][1:]
    if wall_masks[0][0][1] == 0 and wall_masks[0][0][0] != 0:
        wall_masks[0] = wall_masks[0][:1] + [[0,0]] + wall_masks[0][1:]

    # Корекция точек маски правой стены
    if wall_masks[-1][3][1] == 0 and wall_masks[-1][3][0] != width:
        wall_masks[-1] = wall_masks[-1][:3] + [[width,0]] + wall_masks[-1][3:]
    if wall_masks[-1][2][1] == height and wall_masks[-1][2][0] != width:
        wall_masks[-1] = wall_masks[-1][:3] + [[width,height]] + wall_masks[-1][3:]


    # Создание полотна для маски
    image_mask = np.zeros((height, width, 3), dtype="uint8")

    # Отображение маски потолка из полигона
    cv2.fillPoly(image_mask, pts=[np.array(ceiling_mask)], color=colors[0])
    # Отображение маски пола из полигона
    cv2.fillPoly(image_mask, pts=[np.array(floor_mask)], color=colors[1])
    # Отображение масок стен из полигонов
    for i in range(len(wall_masks)):
        wall_color = tuple(int(c) for c in (np.array(colors[2]) + np.array(walls_color_delta)*i)%256)
        cv2.fillPoly(image_mask, pts=[np.array(wall_masks[i])], color=wall_color)

    # Отображение точек потолка
    for i in range(len(ceiling_points)):
        cv2.circle(image_mask,ceiling_points[i],5,(0,255,255),-1)

    # Отображение точек пола
    for i in range(len(floor_points)):
        cv2.circle(image_mask,floor_points[i],5,(0,255,255),-1)

    return image_mask
def texturing_by_mask_color(image, texture, mask, color):
    '''
    Функция заполняет изображение текстурой по цвету маски.
    Изображение, текстура и маска передаются NumPy массивом (cv2.imread)
    Цвет в виде кортежа или списка из трех каналов цветов (BGR если маска загружена cv2.imread)
    '''

    textured_image = image.copy()
    boolean_mask = ((mask[:,:,0] == color[0]) & (mask[:,:,1] == color[1]) & (mask[:,:,2] == color[2]))
    textured_image[boolean_mask] = texture[boolean_mask]
    return textured_image
def texture_transformation(texture, source_points, destination_points, destination_size = (800, 600)):
    '''
    Функция преобразования текстуры по опорным точкам. Работает по 3-м или 4-м точкам. Больше игнорируется.
    texture - NumPy массив текстуры;
    source_points - список исходных точек на текстуре;
    destination_points - список конечных точек на изображении;
    destination_size - размер изображения на которое переносится текстура.
    '''
    # Проверяем длину переданных точек (заодно используем малоизвестный "моржовый оператор")
    if (len_points := min(len(source_points), len(destination_points))) > 3:
        hf, _ = cv2.findHomography(np.float32(source_points[:4]), np.float32(destination_points[:4]))
        warp_texture = cv2.warpPerspective(texture, hf, destination_size)
    elif len_points == 3:
        warp_matrix = cv2.getAffineTransform(np.float32(source_points[:3]), np.float32(destination_points[:3]))
        warp_texture = cv2.warpAffine(texture, warp_matrix, destination_size)
    else:
        warp_texture = np.zeros(destination_size + (3,), dtype="uint8")
    return warp_texture
# Функции определения точек стен
def calc_walls(ceiling_points, floor_points):
    '''
    Функция постоения точек стен по последовательным парам точек пола и потолка.
    Работает от 3-х точек и более. Если один из списков длиннее, то оставшиеся точки игнорируются.
        ceiling_points, floor_points - списки точек пола и потолка.
    '''
    # Создание точек стен
    wall_points = []
    for i in range(0,min(len(ceiling_points), len(floor_points))-1):
        wall = [ceiling_points[i],floor_points[i],floor_points[i+1],ceiling_points[i+1]]
        wall_points.append(wall)
    return wall_points
# Функция расчета дополнительных точек расширенной маски
def add_mask_points(ceiling_points,
                    floor_points,
                    wall_points,
                    size = (800, 600),
                    x_edge = 10,
                    y_edge = 10):
    '''
    Функция добавляет углы изображения к соответствующим опорным точкам.
    Необходима для корректного расчета размера текстуры для заливки.
    ceiling_points, floor_points, wall_points - списки точек пола и потолка;
    wall_points - размер изображения маски;
    '''

    # Извлечение размеров
    width, height = size

    # Создание полигона потолка с добавлением углов изображения
    ceiling_mask = ceiling_points.copy()
    if ceiling_points[0][0] <= x_edge and ceiling_points[0][1] != 0:
        ceiling_mask = [[0,0]] + ceiling_mask
    if ceiling_points[-1][0] >= width - x_edge and ceiling_points[-1][1] != 0:
        ceiling_mask = ceiling_mask + [[width,0]]

    # Создание полигона пола с добавлением углов изображения
    floor_mask = floor_points.copy()
    if floor_points[0][0] <= x_edge and floor_points[0][1] != height:
        floor_mask = [[0,height]] + floor_mask
    if floor_points[-1][0] >= width - x_edge and floor_points[-1][1] != height:
        floor_mask = floor_mask + [[width,height]]

    # Создание полигонов стен с добавлением углов изображения
    wall_masks = []
    for i in range(0,min(len(ceiling_points),len(floor_points))-1):
        wall = [ceiling_points[i],floor_points[i],floor_points[i+1],ceiling_points[i+1]]
        wall_masks.append(wall)

    # Корекция точек маски левой стены
    if wall_masks[0][1][1] >= height - y_edge and wall_masks[0][1][0] != 0:
        wall_masks[0] = wall_masks[0][:1] + [[0,height]] + wall_masks[0][1:]
    if wall_masks[0][0][1] <= y_edge and wall_masks[0][0][0] != 0:
        wall_masks[0] = wall_masks[0][:1] + [[0,0]] + wall_masks[0][1:]

    # Корекция точек маски правой стены
    if wall_masks[-1][3][1] <= y_edge and wall_masks[-1][3][0] != width:
        wall_masks[-1] = wall_masks[-1][:3] + [[width,0]] + wall_masks[-1][3:]
    if wall_masks[-1][2][1] >= height - y_edge and wall_masks[-1][2][0] != width:
        wall_masks[-1] = wall_masks[-1][:3] + [[width,height]] + wall_masks[-1][3:]

    return ceiling_mask, floor_mask, wall_masks
def calc_room_skin_sizes(ceiling_points,
                         floor_points,
                         r_l, r_w, r_h):
    '''
    Функция вычисляет размеры развертки текстур комнаты на основании ее размеров.
      ceiling_points - опорные точки потолка;
      floor_points - опорные точки пола;
      r_l, r_w, r_h - длина, ширина и высота комнаты в см.
    '''

    # Функция для вычисления расстояния между точками через вектора NumPy
    def calc_dist(points):
        square = np.square(np.array(points[0]) - np.array(points[1]))
        sum_square = np.sum(square)
        return  np.sqrt(sum_square)

    # Создание точек стен
    wall_points = calc_walls(ceiling_points, floor_points)

    # Определяем самую длинную видимую сторону
    walls_skin_size = [top_left_w_edge := calc_dist([wall_points[0][0], wall_points[0][3]]),
                       bottom_left_w_edge := calc_dist([wall_points[0][1], wall_points[0][2]]),
                       top_rigth_w_edge := calc_dist([wall_points[-1][0], wall_points[-1][3]]),
                       bottom_rigth_w_edge := calc_dist([wall_points[-1][1], wall_points[-1][2]])]

    max_edge_index = walls_skin_size.index(max(walls_skin_size))
    walls_edges_scale = list(np.array(walls_skin_size)/walls_skin_size[max_edge_index])

   # Вычисление длин ребер потолка
    if (len_points := min(len(ceiling_points), len(floor_points))) == 3:
        ceiling_skin_size = [r_l*walls_edges_scale[0],r_w*walls_edges_scale[2]]
    elif len_points == 4:
        # Вычисляем какая из граней длиннее и в соответствии с этим корректируем вторую
        ceiling_skin_size = [r_l*walls_edges_scale[0], r_w, r_l*walls_edges_scale[2]]
    else:
        print('Меньше 2-х и больше 4-х точек пока не обрабатываем!')
        ceiling_skin_size = []

    # Вычисление длин ребер пола
    if (len_points := min(len(ceiling_points),len(floor_points))) == 3:
        floor_skin_size = [r_l*walls_edges_scale[1],r_w*walls_edges_scale[3]]
    elif len_points == 4:
        floor_skin_size = [r_l*walls_edges_scale[1], r_w, r_l*walls_edges_scale[3]]

    # Вычисление размеров стен
    walls_skin_size = []

    # Вычисление длин ребер левой стены
    walls_skin_size.append([r_l*walls_edges_scale[0], r_h, r_l*walls_edges_scale[1]])

    # Проверка количества стен
    if len_points == 4:
        # Добавляем среднюю сторону
        walls_skin_size.append([r_w, r_h, r_w])
        # Вычисление длин ребер правой стены
        walls_skin_size.append([r_l*walls_edges_scale[2], r_h, r_l*walls_edges_scale[3]])

    elif len_points == 3:
        # Вычисление длин ребер правой стены
        walls_skin_size.append([r_w*walls_edges_scale[2], r_h, r_w*walls_edges_scale[3]])

    return ceiling_skin_size, floor_skin_size, walls_skin_size, wall_points
# Функция расчета точек и отрисовки развертки комнаты
def draw_room_skin(ceiling_skin_size,
                   floor_skin_size,
                   walls_skin_size,
                   r_l, r_w, r_h,
                   colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)],
                   walls_color_delta = (0, 75, 0)):

    '''
    Функция преобразует размеры и отображает развертку.
      ceiling_skin_size - размеры потолка в см;
      floor_skin_size - размеры пола в см;
      walls_skin_size - размеры стен в см;
      r_l, r_w, r_h - длина, ширина и высота комнаты в см;
      colors - список цветов потолка, пола, стен для маски;
      walls_color_delta - смещение цветовой палитры для цвета стен.
    '''

    # Обработка ситуации с 2-мя видимыми стенами
    if (len_points := min(len(ceiling_skin_size), len(floor_skin_size), len(walls_skin_size))) == 2:
        skin_map = np.zeros((int(r_l*2+r_h),int(r_l+r_w),3), dtype="uint8")
        skin_ceiling_points = [[r_l,r_l-ceiling_skin_size[0]],[r_l,r_l],[r_l+ceiling_skin_size[1],r_l]]
        skin_floor_points = [[r_l,r_l+r_h+floor_skin_size[0]],[r_l,r_l+r_h],[r_l+floor_skin_size[1],r_l+r_h]]
        skin_walls_points = []
        skin_walls_points.append([[r_l-walls_skin_size[0][0],r_l],[r_l-walls_skin_size[0][2],r_l+walls_skin_size[0][1]],[r_l,r_l+walls_skin_size[0][1]],[r_l,r_l]])
        skin_walls_points.append([[r_l,r_l],[r_l,r_l+walls_skin_size[1][1]],[r_l+walls_skin_size[1][2],r_l+walls_skin_size[1][1]],[r_l+walls_skin_size[1][0],r_l]])
    # Обработка ситуации с 3-мя видимыми стенами
    elif len_points == 3:
        skin_map = np.zeros((int(r_l*2+r_h),int(r_l*2+r_w),3), dtype="uint8")
        skin_ceiling_points = [[r_l,r_l-ceiling_skin_size[0]],[r_l,r_l],[r_l+r_w,r_l],[r_l+r_w,r_l-ceiling_skin_size[2]]]
        skin_floor_points = [[r_l,r_l+r_h+floor_skin_size[0]],[r_l,r_l+r_h],[r_l+r_w,r_l+r_h],[r_l+r_w,r_l+r_h+floor_skin_size[2]],]
        skin_walls_points = []
        skin_walls_points.append([[r_l-walls_skin_size[0][0],r_l],[r_l-walls_skin_size[0][2],r_l+walls_skin_size[0][1]],[r_l,r_l+walls_skin_size[1][1]],[r_l,r_l]])
        skin_walls_points.append([[r_l,r_l],[r_l,r_l+r_h],[r_l+r_w,r_l+r_h],[r_l+r_w,r_l]])
        skin_walls_points.append([[r_l+r_w,r_l],[r_l+r_w,r_l+walls_skin_size[1][1]],[r_l+r_w+walls_skin_size[2][2],r_l+walls_skin_size[1][1]],[r_l+r_w+walls_skin_size[2][0],r_l]])
    else:
        print('Меньше 2-х и больше 4-х стен пока не обрабатываем!')

    if 'skin_map' in locals():
        # Отображение развертки потолка из полигона
        cv2.fillPoly(skin_map, pts=[np.array(skin_ceiling_points).astype(np.int64)], color=colors[0])
        # Отображение маски пола из полигона
        cv2.fillPoly(skin_map, pts=[np.array(skin_floor_points).astype(np.int64)], color=colors[1])
        # Отображение масок стен из полигонов
        for i in range(len(skin_walls_points)):
            wall_color = tuple(int(c) for c in (np.array(colors[2]) + np.array(walls_color_delta)*i)%256)
            cv2.fillPoly(skin_map, pts=[np.array(skin_walls_points[i]).astype(np.int64)], color=wall_color)

# Функция расчета точек
def calc_room_skin_points(ceiling_skin_size,
                          floor_skin_size,
                          walls_skin_size,
                          r_l, r_w, r_h):

    '''
    Функция преобразует размеры развертки в точки для наложения на текстуру.
      ceiling_skin_size - размеры потолка в см;
      floor_skin_size - размеры пола в см;
      walls_skin_size - размеры стен в см;
      r_l, r_w, r_h - длина, ширина и высота комнаты в см;
      colors - список цветов потолка, пола, стен для маски;
      walls_color_delta - смещение цветовой палитры для цвета стен.
    '''

    # Обработка ситуации с 2-мя видимыми стенами
    if (len_points := min(len(ceiling_skin_size), len(floor_skin_size), len(walls_skin_size))) == 2:
        skin_ceiling_points = [[r_l,r_l-ceiling_skin_size[0]],[r_l,r_l],[r_l+ceiling_skin_size[1],r_l]]
        skin_floor_points = [[r_l,r_l+r_h+floor_skin_size[0]],[r_l,r_l+r_h],[r_l+floor_skin_size[1],r_l+r_h]]
        skin_walls_points = []
        skin_walls_points.append([[r_l-walls_skin_size[0][0],r_l],[r_l-walls_skin_size[0][2],r_l+walls_skin_size[0][1]],[r_l,r_l+walls_skin_size[0][1]],[r_l,r_l]])
        skin_walls_points.append([[r_l,r_l],[r_l,r_l+walls_skin_size[1][1]],[r_l+walls_skin_size[1][2],r_l+walls_skin_size[1][1]],[r_l+walls_skin_size[1][0],r_l]])
    # Обработка ситуации с 3-мя видимыми стенами
    elif len_points == 3:
        skin_ceiling_points = [[r_l,r_l-ceiling_skin_size[0]],[r_l,r_l],[r_l+r_w,r_l],[r_l+r_w,r_l-ceiling_skin_size[2]]]
        skin_floor_points = [[r_l,r_l+r_h+floor_skin_size[0]],[r_l,r_l+r_h],[r_l+r_w,r_l+r_h],[r_l+r_w,r_l+r_h+floor_skin_size[2]],]
        skin_walls_points = []
        skin_walls_points.append([[r_l-walls_skin_size[0][0],r_l],[r_l-walls_skin_size[0][2],r_l+walls_skin_size[0][1]],[r_l,r_l+walls_skin_size[1][1]],[r_l,r_l]])
        skin_walls_points.append([[r_l,r_l],[r_l,r_l+r_h],[r_l+r_w,r_l+r_h],[r_l+r_w,r_l]])
        skin_walls_points.append([[r_l+r_w,r_l],[r_l+r_w,r_l+walls_skin_size[1][1]],[r_l+r_w+walls_skin_size[2][2],r_l+walls_skin_size[1][1]],[r_l+r_w+walls_skin_size[2][0],r_l]])
    else:
        print('Меньше 2-х и больше 4-х стен пока не обрабатываем!')
        return [], [], []

    return skin_ceiling_points, skin_floor_points, skin_walls_points
def skin_recalculator(points, mask, skin_points):

    '''
    Функция переносит дополнительные точки маски на развертку, создавая дополненную развертку.
       points - список точек элемента маски;
       mask - дополненный список точек элементов маски;
       skin_points - точки развертки.
    '''

    if len(points) < len(mask):
        # Проверяем длину переданных точек (заодно используем малоизвестный "моржовый оператор")
        if (len_points := len(points)) > 3:
            h_matrix, _ = cv2.findHomography(np.float32(points[:4]), np.float32(skin_points[:4]))
            transformed_pts = np.int16(cv2.perspectiveTransform(np.float32(mask).reshape(-1, 1, 2), h_matrix).reshape(np.float32(mask).shape))
            return transformed_pts.tolist()
        elif len_points == 3:
            a_matrix = cv2.getAffineTransform(np.float32(points[:3]), np.float32(skin_points[:3]))
            transformed_pts = []
            for point in mask:
              #print(np.dot(a_matrix,[[point[0]],[point[1]],[1]]))
              transformed_pts.append(np.dot(a_matrix,[[point[0]],[point[1]],[1]]).tolist())
            return transformed_pts
    else:
        return skin_points
def room_skin_corrector(ceiling_points, floor_points, wall_points,
                        ceiling_mask, floor_mask, wall_masks,
                        skin_ceiling_points, skin_floor_points, skin_walls_points):

    '''
    Функция единовременной обработки всех масок и разверток.
    '''

    re_skin_ceiling_points = skin_recalculator(ceiling_points, ceiling_mask, skin_ceiling_points)
    re_skin_floor_points = skin_recalculator(floor_points, floor_mask, skin_floor_points)
    re_skin_walls_points = []
    for i in range(len(wall_points)):
        re_skin_walls_points.append(skin_recalculator(wall_points[i], wall_masks[i], skin_walls_points[i]))

    return re_skin_ceiling_points, re_skin_floor_points, re_skin_walls_points
# Функция поворота и масштабирования координат
def coors_rotation(coords, center = (500, 300), angle = 90, scale = 1):
  '''
  Функция пороворота и масштабирования координат.
    coords - список координат;
    center - центр вращения;
    angle - угол поворота в градусах;
    scale - коэффициент масштабирования.
  '''

  rot_mat = cv2.getRotationMatrix2D(center, angle, scale)

  transformed_coords = []

  for c in coords:
    transformed_coords.append(np.dot(rot_mat, np.array([c[0],c[1],1])).tolist())

  return transformed_coords
# Функция сдвига точек
def coors_moving(coords, dx, dy):
  '''
  Функция сдвига координат.
    coords - спискок координат;
    dx - сдвиг по оси х;
    dy - сдвиг по оси y.
  '''

  # Если одноуровневый спискок координат
  if len(coords[0]) == 2 and type(coords[0][0]) == list:
      coords_np = np.array(coords)
      coord_np = coords_np + [dx, dy]
      return coord_np.tolist()
  else:
      new_coords = []
      for coords_part in coords:
        coords_part_np = np.array(coords_part)
        coord_part_np = coords_part_np + [dx, dy]
        new_coords.append(coord_part_np.tolist())
      return new_coords
# Определение максимального и минимального значения координат
def calc_xy_limits(coords):

    '''
    Функция для вычисления максимальных и минимальных значения по осям X и Y
    '''
    all_xy = []

    for level_1 in coords:
        for level_2 in level_1:
            if type(level_2) == list:
                for level_3 in level_2:
                    all_xy.append(level_3)
            else:
                all_xy.append(level_2)
    all_x = all_xy[::2]
    all_y = all_xy[1::2]
    return [min(all_x),min(all_y),max(all_x),max(all_y)]
# Вычисление требуемого размера текстуры
def prepare_texture(points, json_path, custom_shift = None, herringbone = None, dx = 0, dy = 0, rescale = None):

    '''
      Функция подготовки текстуры нужных размеров для отображения точек развертки.
      Для обеспечения возможности поворота, вычисляется диагональ.
      Есть возможность задать дополнительный запас по осям для смещения.
      Функция возвращает созданную текстуру, а также коэффициент масштабирования точек развертки и необходимое смещение.
    '''

    x_min, y_min, x_max, y_max = calc_xy_limits(points)
    min_size = ((x_max-x_min)**2+(y_max-y_min)**2)**0.5
    # Создание подходящего полотна
    texture_field = create_texture_canvas(min_size+dx, min_size+dy, json_path, custom_shift, herringbone, rescale)

    # Вычисление коэффициента масштабирования
    texture_size = texture_field.shape[0]
    scale_coef = texture_size/min_size

    # Вычисление смещения центра точек в начало координат
    points_dx = (x_max+x_min)/2
    points_dy = (y_max+y_min)/2

    return texture_field, texture_size, scale_coef, points_dx, points_dy

def build_sceleton_points(points, angle, texture_size, scale_coef, dx, dy):

    '''
    Функция пересчета точек развертки в координаты пикселей на текстуре с учетом подготовленной текстуры
      texture_field - NumPy массив текстуры (используется для пересчета масштаба точек);
      points - точки развертки;
      angle - угол поворота в градусах (вращается развертка, текстура не подвижна).
    '''

    new_points = coors_moving(points, texture_size//2-dx, texture_size//2-dy)

    if len(new_points[0]) == 2:
        scaled_points = coors_rotation(new_points, (texture_size//2, texture_size//2), angle, scale_coef)
    else:
        scaled_points = []
        for p in new_points:
            scaled_points.append(coors_rotation(p, (texture_size//2, texture_size//2), angle, scale_coef))
    return scaled_points
def brith_mask(image, texture, seg_mask, blender_image, surface):
    #seg_mask = seg_mask[:,:,0]
    seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[np.where(seg_mask==0)] = 0
    gray_tex = cv2.cvtColor(texture, cv2.COLOR_RGB2GRAY)
    Gaussian = cv2.GaussianBlur(gray, (101, 101), 0)
    hist = cv2.calcHist([Gaussian], [0], None,  [256], [20, 256])
    Gaussian_tex = cv2.GaussianBlur(gray_tex, (51, 51), 0) 
    hist_tex = cv2.calcHist([Gaussian_tex], [0], None,  [256], [10, 256])
    if np.argmax(hist_tex) < np.argmax(hist)+20:
        const = np.argmax(hist)  - (np.argmax(hist) - np.argmax(hist_tex)) 
    elif np.argmax(hist)+20 < np.argmax(hist_tex):
        const = np.argmax(hist) +20
    #const = np.argmax(hist)
    shadow = cv2.GaussianBlur(gray, (55, 55), 0) 
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
def gaus_img(img_cont, seg_mask):
    contours = []
    gray = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(img_cont, (11, 11), 0)
    contours, _ = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_cont = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask_cont, contours, -1, (255,255,255), 3)
    img_cont[np.where(mask_cont>0)] = blurred_img[np.where(mask_cont>0)]
    
    return img_cont
def texture_transformation_1(original_image, seg_all,
                            room_l, room_w, room_h,
                            ceiling_points, floor_points,
                            angle_f, angle_w, angle_c,
                            json_list, surface,
                            h_f, h_w, h_c,
                            s_f, s_w, s_c,
                            rescale = 1.):
    
    mask = build_mask(ceiling_points,
               floor_points,
               size = original_image.shape[::-1][1:])
    textured_mask = mask.copy()           
    walls_points = calc_walls(ceiling_points, floor_points)
    ceiling_skin_size, floor_skin_size, walls_skin_size, wall_points = calc_room_skin_sizes(ceiling_points, floor_points, room_l, room_w, room_h)
    ceiling_mask, floor_mask, wall_masks = add_mask_points(ceiling_points, floor_points, wall_points, size = (mask.shape[1], mask.shape[0]))
    skin_ceiling_points, skin_floor_points, skin_walls_points = calc_room_skin_points(ceiling_skin_size, floor_skin_size, walls_skin_size, room_l, room_w, room_h)
    re_skin_ceiling_points, re_skin_floor_points, re_skin_walls_points = room_skin_corrector(ceiling_points, floor_points, wall_points, ceiling_mask, floor_mask, wall_masks, skin_ceiling_points, skin_floor_points, skin_walls_points)
    if json_list[1] is not None and surface[1] == True:
        texture_field, texture_size, scale_coef, points_dx, points_dy = prepare_texture(re_skin_walls_points, json_list[1], herringbone = h_w, rescale = rescale)
        scaled_points = build_sceleton_points(skin_walls_points, angle_w, texture_size, scale_coef, points_dx, points_dy)
        for i in range(len(wall_points)):
            transformed_texture = texture_transformation(texture_field, scaled_points[i], walls_points[i], destination_size = mask.shape[::-1][1:])
            textured_mask = texturing_by_mask_color(textured_mask, transformed_texture, mask, (np.array((255,0,0)) + np.array((0,75,0))*i)%256)
    if json_list[0] is not None and surface[0] == True:
        texture_field, texture_size, scale_coef, points_dx, points_dy = prepare_texture(re_skin_floor_points, json_list[0], herringbone = h_f, rescale = rescale)
        scaled_points = build_sceleton_points(skin_floor_points, angle_f, texture_size, scale_coef, points_dx, points_dy)

        transformed_texture = texture_transformation(texture_field, scaled_points, floor_points, destination_size = mask.shape[::-1][1:])
        textured_mask = texturing_by_mask_color(textured_mask, transformed_texture, mask, (0,0,255))

    if json_list[2] is not None and surface[2] == True:
        texture_field, texture_size, scale_coef, points_dx, points_dy = prepare_texture(re_skin_ceiling_points, json_list[2], herringbone = h_c, rescale = rescale)
        scaled_points = build_sceleton_points(skin_ceiling_points, angle_c, texture_size, scale_coef, points_dx, points_dy)

        transformed_texture = texture_transformation(texture_field, scaled_points, ceiling_points, destination_size = mask.shape[::-1][1:])
        textured_mask = texturing_by_mask_color(textured_mask, transformed_texture, mask, (0,255,0))
    output_mask = cv2.cvtColor(textured_mask, cv2.COLOR_BGR2RGB)
    output_mask_floor = output_mask.copy()
    output_mask_wall = output_mask.copy()
    output_mask_ceiling = output_mask.copy()
    output_mask_floor[np.where(output_mask[:,:,0] == 255)] = 0
    output_mask_floor[np.where(output_mask[:,:,2] == 255)] = 0
    output_mask_wall[np.where(output_mask[:,:,1] == 255)] = 0
    output_mask_wall[np.where(output_mask[:,:,2] == 255)] = 0
    output_mask_ceiling[np.where(output_mask[:,:,0] == 255)] = 0
    output_mask_ceiling[np.where(output_mask[:,:,2] == 255)] = 0
    output_mask_floor[np.where(seg_all[0]==0)] = 0
    output_mask_wall[np.where(seg_all[1]==0)] = 0
    output_mask_ceiling[np.where(seg_all[2]==0)] = 0
    output_image = original_image.copy()
    #output_image[np.where(seg_all[0]>0)] = output_mask[np.where(seg_all[0]>0)]
    if surface[0] == True:
        output_image[np.where(output_mask_floor>0)] = output_mask[np.where(output_mask_floor>0)]
        tex_mask = np.zeros((output_image.shape[0],output_image.shape[1], 3), dtype=np.uint8)
        tex_mask[np.where(output_mask_floor>0)] = output_mask[np.where(output_mask_floor>0)]
        if s_f == True:
            output_image = brith_mask(original_image, tex_mask, seg_all[0], output_image, surface = 'floor')
        gaus_img(output_image, seg_all[0])
    if surface[2] == True:
        output_image[np.where(output_mask_ceiling>0)] = output_mask[np.where(output_mask_ceiling>0)]
        tex_mask = np.zeros((output_image.shape[0],output_image.shape[1], 3), dtype=np.uint8)
        tex_mask[np.where(output_mask_ceiling>0)] = output_mask[np.where(output_mask_ceiling>0)]
        if s_c == True:
            output_image = brith_mask(original_image, tex_mask, seg_all[2], output_image, surface = 'ceiling')
        gaus_img(output_image, seg_all[2])
    if surface[1] == True:
        output_image[np.where(output_mask_wall>0)] = output_mask[np.where(output_mask_wall>0)]
        tex_mask = np.zeros((output_image.shape[0],output_image.shape[1], 3), dtype=np.uint8)
        tex_mask[np.where(output_mask_wall>0)] = output_mask[np.where(output_mask_wall>0)]
        if s_w == True:
            output_image = brith_mask(original_image, tex_mask, seg_all[1], output_image, surface = 'wall')
        gaus_img(output_image, seg_all[1])
    return output_image