from utils.datasets import LoadImagesAndLabels
import json
import os

# coco的json编码格式，整个数据集共用一个文件
# 需要预处理images列表，找到id 到file_name的映射
# annotations列表 每个annotation结构体
# image_id 表示对应的image_id，可以查到file_name
# category_id 表示类别编号
# bbox有4ge数值
# 左上角x 左上角y 框宽度 框高度

# xml的编码格式，文件名与图像相同
# 

# txt的编码格式，文件名与图像相同
# n行，每行5列，一行代表一个目标框
# 类别编号 中心点x 中心点y 框宽度 框高度

# 划分训练测试集方法，划分目录，或者在在图像统一目录下增加txt文件

# coco
def process_coco(json_file, img_dir):
    image_id_map = {}
    id_name_map = {}
    id_size_map = {}
    file_label_map = {}
    with open(json_file, 'r+') as f:
        j = json.load(f)
        images = j['images']
        labels = j['annotations']
        categories = j['categories']
    for img in images:
        image_id_map[img['id']] = img['file_name']
        id_size_map[img['id']] = (img['width'], img['height'])
        # if (len(image_id_map) == 1) :
        #     print(image_id_map)
    for cate in categories:
        id_name_map[cate['id']] = cate['supercategory']
    for label in labels:
        file_name = image_id_map.get(label['image_id'])
        if file_name is None:
            print('image not find id:', label['image_id'])
            continue
        obj_id = label['category_id']
        x, y, w, h = label['bbox']
        x_c = x + w / 2
        y_c = y + h / 2
        i_w, i_h = id_size_map[label['image_id']]
        if file_label_map.get(file_name) is None:
            file_label_map[file_name] = [[obj_id, x_c/i_w, y_c/i_h, w/i_w, h/i_h]]
        else:
            file_label_map[file_name].append([obj_id, x_c/i_w, y_c/i_h, w/i_w, h/i_h])
    for file_name, boxes in file_label_map.items():
        file_path = os.path.join(img_dir, file_name.split('.')[0]+'.txt')
        # print(file_path)
        # for box in boxes:
        #     print('{} {} {} {} {}\n'.format(*box))
        with open(file_path, 'w+') as f:
            for box in boxes:
                f.write('{} {} {} {} {}\n'.format(*box))

# process_coco('dataset/coco/annotations/instances_val2017.json', 'dataset/coco/val2017')
# process_coco('dataset/coco/annotations/instances_train2017.json', 'dataset/coco/train2017')

LoadImagesAndLabels('dataset/coco/train2017',
                    img_size=640,
                    batch_size=16,
                    augment=False, # mosaic and other augment
                    hyp=None, #
                    rect=False,
                    image_weights=False,
                    cache_images=False,
                    single_cls=False,
                    stride=32,
                    pad=0)

# neu