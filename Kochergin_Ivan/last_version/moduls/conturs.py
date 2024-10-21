import os
import requests
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import matplotlib.colors as mcolors
import os
from scipy.spatial import ConvexHull
# Функция для проверки, лежит ли точка на границе bounding box
# +
def is_on_bbox(point, bbox):
    x, y, w, h = bbox
    return (point[0] == x or point[0] == x + w or
            point[1] == y or point[1] == y + h)
# +
def remove_redundant_points(contour):
    contour = np.array(contour)  # Преобразуем список в numpy массив
    if len(contour) < 3:
        return contour  # Ничего не делать, если меньше трех точек
    # Список для хранения новых точек
    new_contour = [contour[0]]  # Первая точка всегда добавляется
    for i in range(1, len(contour) - 1):
        p1 = contour[i - 1]
        p2 = contour[i]
        p3 = contour[i + 1]
        # Проверяем, являются ли p1, p2 и p3 коллинеарными
        if not is_collinear(p1, p2, p3):
            new_contour.append(p2)  # Добавляем точку, если она не средняя
    new_contour.append(contour[-1])  # Последняя точка всегда добавляется
    return np.array(new_contour)
# +
def is_collinear(p1, p2, p3):
    # Проверяем коллинеарность с помощью площади треугольника
    return np.isclose((p2[1] - p1[1]) * (p3[0] - p2[0]), (p3[1] - p2[1]) * (p2[0] - p1[0]))
# +
def close_contour(contour):
    # Проверяем, замкнут ли контур
    if not np.array_equal(contour[0], contour[-1]):
        # Если не замкнут, добавляем первую точку в конец
        contour = np.vstack([contour, contour[0]])
    return contour
# +
def inpaint_color_mask(mask):
    # Преобразование маски в оттенки серого для создания маски инпейнтинга
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Создание бинарной маски, где белые области - это дыры, которые нужно заполнить
    _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY_INV)
    # Применение инпейнтинга
    inpainted_mask = cv2.inpaint(mask, binary_mask, 3, cv2.INPAINT_TELEA)
    return inpainted_mask
# +
def extract_wall_mask(image):
    # Синий канал
    #wall_mask = image.copy()
    #plt.imshow(image)
    #plt.show()
    floor_mask = (image[:, :, 0] >= 200)
    ceiling_mask = (image[:, :, 1] >= 200)
    wall_mask = (image[:, :, 2] >= 200)
    #wall_mask[wall_mask[:, :, 2] != 100] = 0
    return wall_mask, floor_mask, ceiling_mask
# +
def is_point_on_segment(point, p1, p2):
    """
    Проверяет, лежит ли точка на отрезке, соединяющем точки p1 и p2.
    :param point: Проверяемая точка [x, y]
    :param p1: Первая точка отрезка [x, y]
    :param p2: Вторая точка отрезка [x, y]
    :return: True, если точка лежит на отрезке, иначе False
    """
    # Проверяем, что точка находится между p1 и p2 по обеим координатам x и y
    if min(p1[0], p2[0]) <= point[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= point[1] <= max(p1[1], p2[1]):
        # Вычисляем векторные произведения для проверки коллинеарности
        cross_product = (point[1] - p1[1]) * (p2[0] - p1[0]) - (point[0] - p1[0]) * (p2[1] - p1[1])
        if abs(cross_product) < 1e-6:  # допуск для вычислений с плавающей точкой
            return True
    return False
# +
def douglas_peucker(contour, max_vertices):
    epsilon = 0.01  # Начальное значение, будет увеличиваться до достижения max_vertices
    while True:
        approx_contour = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx_contour) <= max_vertices:
            #print(epsilon)
            return approx_contour
        epsilon += 0.01
# Функция для проверки, должна ли точка быть вставлена между двумя точками упрощенного контура
# +
def should_insert_between(p1, p2, point):
    within_bbox = (min(p1[0], p2[0]) <= point[0] <= max(p1[0], p2[0]) and
                    min(p1[1], p2[1]) <= point[1] <= max(p1[1], p2[1]))
    return within_bbox
# +
def douglas_peucker_refine(contour, simplified_contour):
    simplified_contour=close_contour(simplified_contour)
    # Получаем bounding box для исходного контура
    bbox = cv2.boundingRect(contour)
    # Извлекаем все точки на bounding box
    points_on_bbox = [point for point in contour if is_on_bbox(point, bbox)]
    #visualize_points_on_contour(contour, points_on_bbox)
    # Вставляем точки на bounding box в упрощенный контур
    for point in points_on_bbox:
        for i in range(len(simplified_contour) - 1):
            if is_point_on_segment(point, simplified_contour[i], simplified_contour[i + 1]):
                break  # Если точка уже покрыта отрезком, выходим из цикла
        else:  # Если точка не покрыта ни одним сегментом, вставляем
            for i in range(len(simplified_contour) - 1):
                if should_insert_between(simplified_contour[i], simplified_contour[i + 1], point):
                    simplified_contour = np.insert(simplified_contour, i+1, [point], axis=0)
                    break
    simplified_contour = remove_redundant_points(simplified_contour)
    return simplified_contour
# +
def find_and_simplify_contour(mask, max_vertices ):
    # Преобразование маски к типу данных uint8
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    contour = largest_contour.reshape(-1, 2)
    simplified_contour = douglas_peucker(contour, max_vertices)
    simplified_contour = np.squeeze(simplified_contour)
    # Добавляем точки на bbox, пропущенные Дугласом Пекером
    simplified_contour = douglas_peucker_refine(contour, simplified_contour)
    # Замыкание контура
    if not np.array_equal(simplified_contour[0], simplified_contour[-1]):
        simplified_contour = np.vstack([simplified_contour, simplified_contour[0]])
    return simplified_contour, contour

# +
def get_contour_by_mask (original_image,seg_map):
    # Создадим упрощенную маску сегментации для более эффективного импайнтинга
    #simplified_color_segmentation = get_simplified_color_segmentation(seg_map)
    max_vertices = 11
    # Получаем залитую и упрощенную маску
    inpainted_mask = inpaint_color_mask(seg_map)
    #print(inpainted_mask[200, 200])
    # Выделяем маску стены
    #inpainted_wall_mask = extract_wall_mask(inpainted_mask)
    wall_mask, floor_mask, ceiling_mask = extract_wall_mask(inpainted_mask)
    #plt.imshow(inpainted_wall_mask)
    #plt.show()
    # Поиск и упрощение контура
    #simplified_contour, contour = find_and_simplify_contour(inpainted_wall_mask,max_vertices)
    direct = {'floor': 1,
              'wall': 1,
              'ceiling': 1
              }

    try:
        simplified_contour_f, contour_f = find_and_simplify_contour(floor_mask,max_vertices)
    except:
        direct['floor'] = 0
        simplified_contour_f, contour_f = [], []
    try:
        simplified_contour_w, contour_w = find_and_simplify_contour(wall_mask,max_vertices)
    except:
        direct['wall'] = 0
        simplified_contour_w, contour_w = [], []
    try:
        simplified_contour_c, contour_c = find_and_simplify_contour(ceiling_mask,max_vertices)
    except:
        direct['ceiling'] = 0
        simplified_contour_c, contour_c = [], []
    #return simplified_contour, inpainted_wall_mask, contour
    return simplified_contour_f, contour_f, simplified_contour_w, contour_w, simplified_contour_c, contour_c, direct
# +
def simplify_contour(contour):
    """
    Упрощает контур, удаляя первую и последнюю точки, если они участвуют
    в горизонтальном отрезке длиной более 5 единиц.
    """
    #print(f"Первые три точки контура: {contour[:3]}")
    #print(f"Последние три точки контура:{contour[-3:]}")

    if len(contour) < 2:
        return contour  # Возвращаем контур как есть, если в нём меньше двух точек

    # Проверяем первую точку
    if contour[1][1] == contour[0][1] and abs(contour[1][0] - contour[0][0]) > 5:
        #print("Удаление первой точки:", contour[0])
        contour = contour[1:]  # Удаляем первую точку, если условие выполнено

    # Проверяем последнюю точку
    if len(contour) > 1 and contour[-1][1] == contour[-2][1] and abs(contour[-1][0] - contour[-2][0]) > 5:
        #print("Удаление последней точки:", contour[-1])
        contour = contour[:-1]  # Удаляем последнюю точку, если условие выполнено

    return contour

# +
def adjust_contour(contour):
    """
    Корректирует контур, перемещая точки на основании заданных условий расстояния.

    1. Если расстояние от первой до второй точки контура больше, чем расстояние от первой до последней,
       то переставить первую точку в конец контура.
    2. Если расстояние от последней точки до предпоследней больше, чем от последней до первой,
       то переставляем последнюю точку на первое место контура.
    """
    # Проверяем условия для первой точки
    if np.linalg.norm(contour[0] - contour[1]) > np.linalg.norm(contour[0] - contour[-1]):
        # Перемещаем первую точку в конец
        contour = np.append(contour[1:], [contour[0]], axis=0)

    # Проверяем условия для последней точки после возможной коррекции первой точки
    if np.linalg.norm(contour[-1] - contour[-2]) > np.linalg.norm(contour[-1] - contour[0]):
        # Перемещаем последнюю точку на первое место
        contour = np.append([contour[-1]], contour[:-1], axis=0)

    return contour
# +
def get_floor_ceiling_hulls_1(contour):
    """
    Обрабатывает контур, разделяет его на пол и потолок, создает выпуклые оболочки,
    и возвращает их как списки точек.
    """
    mean_y = (np.max(contour[:, 1]) + np.min(contour[:, 1])) / 2
    floor_contour = contour[contour[:, 1] > mean_y]
    ceiling_contour = contour[contour[:, 1] < mean_y]

    # Адаптируем потолочный контур если начальная точка совпадает с первой точкой исходного контура
    floor_contour = adjust_contour(floor_contour)
    ceiling_contour = adjust_contour(ceiling_contour)

    # Упрощение контуров
    floor_contour = simplify_contour(floor_contour)
    ceiling_contour = simplify_contour(ceiling_contour)

    floor_hull = None
    ceiling_hull = None

    # Создаем выпуклые оболочки, если это возможно
    if len(floor_contour) > 2:
        floor_hull = ConvexHull(floor_contour)
        floor_hull = floor_contour[floor_hull.vertices]
    if len(ceiling_contour) > 2:
        ceiling_hull = ConvexHull(ceiling_contour)
        ceiling_hull = ceiling_contour[ceiling_hull.vertices]

    return floor_hull, ceiling_hull

def get_floor_ceiling_hulls(contour_f, contour_c, direct):
    #mean_y = (np.max(contour[:, 1]) + np.min(contour[:, 1])) / 2
    if direct['floor'] == 1:
        floor_contour = contour_f
        floor_contour = adjust_contour(floor_contour)
        floor_contour = simplify_contour(floor_contour)
        floor_hull = None
        if len(floor_contour) > 2:
            floor_hull = ConvexHull(floor_contour)
            floor_hull = floor_contour[floor_hull.vertices]
    else:
        floor_hull = None
    if direct['ceiling'] == 1:
        ceiling_contour = contour_c
        ceiling_contour = adjust_contour(ceiling_contour)
        ceiling_contour = simplify_contour(ceiling_contour)
        ceiling_hull = None
        if len(ceiling_contour) > 2:
            ceiling_hull = ConvexHull(ceiling_contour)
            ceiling_hull = ceiling_contour[ceiling_hull.vertices]
    else:
        ceiling_hull = None
    return floor_hull, ceiling_hull, direct
def conturs_on_image(original_image,seg_map):
    # Упрощаем контур
    _, contour_f, _, contour_w, simplified_contour, contour_c, direct = get_contour_by_mask(original_image,seg_map)
    # Визуализируем контур
    org_img = original_image.copy()
    floor_contour, ceiling_contour, direct = get_floor_ceiling_hulls(contour_f, contour_c, direct)
    visualized_contour = cv2.drawContours(org_img, [floor_contour], -1, (255, 0, 255), 5)
    visualized_contour = cv2.drawContours(org_img, [ceiling_contour], -1, (255, 255, 0), 5)
    
    return visualized_contour, simplified_contour, direct