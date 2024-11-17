import glob
import os
import cv2
import random
import numpy as np
from pathlib import Path

def get_normal(img_dir, train_txt, normal_dir):
    with open(train_txt, 'r+') as f:
        train_files = f.readlines()
    img_paths = [_.replace('./', img_dir+'/')[:-1] for _ in train_files]
    for img_path in img_paths:
        label_path = img_path.replace('.jpg', '.txt')
        with open(label_path, 'r+') as f:
            labels = f.readlines()

        img = cv2.imread(img_path)
        ih, iw = img.shape[:2]
        abns = []
        for label in labels:
            _, x, y, w, h = [float(_) for _ in label.split()]
            x = int((x - w/2) * iw)
            y = int((y - h/2) * ih)
            w = int(w * iw)
            h = int(h * ih)
            abns.append([x,y,x+w,y+h])
        
        x_min, y_min = iw, ih
        x_max, y_max = 0, 0
        for abn in abns:
            x_min = min(x_min, abn[0])
            y_min = min(y_min, abn[1])
            x_max = max(x_max, abn[2])
            y_max = max(y_max, abn[3])
        
        for i, (x, y, w, h) in enumerate(zip((0, 0, x_max, 0), (0, 0, 0, y_max), (x_min, iw, iw-x_max, iw), (ih, y_min, ih, ih-y_max))):
            if w < 0.5*iw and h < 0.5*ih:
                continue
            if w < 0.3*iw or h < 0.3*ih:
                continue
            nom = img[y:y+h, x:x+w]
            nom_path = os.path.join(normal_dir, img_path.split('/')[-1].split('.')[0] + '_p_{:1d}.jpg'.format(i))
            print(nom_path, nom.shape)
            cv2.imwrite(nom_path, nom)

# get_normal('dataset/neu/IMAGES', 'dataset/neu/IMAGES/train.txt', 'dataset/neu/nom')

def get_normal_scaled(img_dir, train_txt, normal_dir):
    with open(train_txt, 'r+') as f:
        train_files = f.readlines()
    img_paths = [_.replace('./', img_dir+'/')[:-1] for _ in train_files]
    for img_path in img_paths:
        label_path = img_path.replace('.jpg', '.txt')
        with open(label_path, 'r+') as f:
            labels = f.readlines()

        img = cv2.imread(img_path)
        ih, iw = img.shape[:2]
        abns = []
        for label in labels:
            _, x, y, w, h = [float(_) for _ in label.split()]
            x = int((x - w/2) * iw)
            y = int((y - h/2) * ih)
            w = int(w * iw)
            h = int(h * ih)
            abns.append([x,y,x+w,y+h])
        
        x_min, y_min = iw, ih
        x_max, y_max = 0, 0
        for abn in abns:
            x_min = min(x_min, abn[0])
            y_min = min(y_min, abn[1])
            x_max = max(x_max, abn[2])
            y_max = max(y_max, abn[3])
        
        for i, (x, y, w, h) in enumerate(zip((0, 0, x_max, 0), (0, 0, 0, y_max), (x_min, iw, iw-x_max, iw), (ih, y_min, ih, ih-y_max))):
            if w < 0.5*iw and h < 0.5*ih:
                continue
            if w < 0.3*iw or h < 0.3*ih:
                continue
            nom = img[y:y+h, x:x+w]
            nom_path = os.path.join(normal_dir, img_path.split('/')[-1].split('.')[0] + '_p_{:1d}.jpg'.format(i))
            print(nom_path, nom.shape)
            nom = cv2.resize(nom, (iw, ih))
            cv2.imwrite(nom_path, nom)

# get_normal_scaled('dataset/neu/IMAGES', 'dataset/neu/IMAGES/train.txt', 'dataset/neu/nom-scaled')

def get_abnormal(img_dir, train_txt, abnormal_dir):
    with open(train_txt, 'r+') as f:
        train_files = f.readlines()
    img_paths = [_.replace('./', img_dir+'/')[:-1] for _ in train_files]
    for img_path in img_paths:
        label_path = img_path.replace('.jpg', '.txt')
        with open(label_path, 'r+') as f:
            labels = f.readlines()

        img = cv2.imread(img_path)
        ih, iw = img.shape[:2]
        for i, label in enumerate(labels):
            obj, x, y, w, h = [float(_) for _ in label.split()]
            x = int((x - w/2) * iw)
            y = int((y - h/2) * ih)
            w = int(w * iw)
            h = int(h * ih)
            abn = img[y:y+h, x:x+w]
            abn_path = os.path.join(abnormal_dir, img_path.split('/')[-1].split('.')[0] + '_abn_{:02d}_{:1d}.jpg'.format(i, int(obj)))
            print(abn_path)
            cv2.imwrite(abn_path, abn)

# get_abnormal('dataset/neu/IMAGES', 'dataset/neu/IMAGES/train.txt', 'dataset/neu/abn')

def generate_abnormal(normal_dir, abnormal_dir, out_dir, size=1500, num=3, thresh=2, gamma=1.5):
    nom_paths = glob.glob(normal_dir + '/*.jpg')
    abn_paths = glob.glob(abnormal_dir + '/*.jpg')

    nom_dir = os.path.join(out_dir, 'normal')
    abnormal_count = 0

    while abnormal_count < size:

        gen_name = 'gen_{:04d}.jpg'.format(abnormal_count)
        gen_path = os.path.join(out_dir, gen_name)
        label_path = os.path.join(out_dir, gen_name.replace('.jpg', '.txt'))
        nom_path = os.path.join(nom_dir, gen_name)

        # pick nom
        nom = random.choice(nom_paths)
        nom = cv2.imread(nom)

        # pick targets
        nt = random.randint(1,num)
        abns = random.choices(abn_paths ,k=nt)
        abns = [cv2.imread(_) for _ in abns]

        if nt == 0:
            continue

        # random flip and rotate targets
        abns = [cv2.flip(_, 0) if random.random() > 0.5 else _ for _ in abns]
        abns = [cv2.flip(_, 1) if random.random() > 0.5 else _ for _ in abns]
        abns = [cv2.rotate(_, cv2.ROTATE_90_CLOCKWISE) if random.random() > 0.5 else _ for _ in abns]

        # nonoverlapping blurring put targets
        nom_copy = np.copy(nom)
        ih, iw = nom.shape[:2]
        
        occupied = []
        mean_val = nom.mean();
        for abn in abns:
            h, w = abn.shape[:2]

            # base filter
            if max(abn.mean()/mean_val, mean_val/abn.mean()) > thresh:
                continue
            if h > ih or w > iw:
                continue

            # occup mask
            offset = [int(0.3*h), int(0.3*w)]
            avail_mask = np.ones((ih-int(0.4*h), iw-int(0.4*w)), dtype=bool)
            ah, aw = avail_mask.shape
            for (xmin, ymin, xmax, ymax) in occupied:
                occup_ymin =  min(max(ymin+offset[0]-h, 0), ah)
                occup_ymax = min(ymax+offset[0], ah)
                occup_xmin = min(max(xmin+offset[1]-w, 0), aw)
                occup_xmax = min(xmax+offset[1], aw)
                avail_mask[occup_ymin:occup_ymax, occup_xmin:occup_xmax] = False
            avail_place = np.argwhere(avail_mask)
            if len(avail_place) == 0:
                continue

            # choose place
            choose = random.choice(avail_place)
            nmo_coord = [max(choose[1]-offset[1],0), max(choose[0]-offset[0],0)]
            nmo_coord.extend([min(iw-nmo_coord[0], w+min(choose[1]-offset[1], 0)), min(ih-nmo_coord[1], h+min(choose[0]-offset[0], 0))])
            abn_coord = [-min(choose[1]-offset[1], 0), -min(choose[0]-offset[0],0), *nmo_coord[2:]]
            # print('shape', nom.shape, abn.shape)
            # print('choose', choose)
            # print('offset', int(0.7*h), int(0.7*w))
            # print('nom_coord', choose[0]-offset[0], choose[1]-offset[1])

            # gamma mask
            mask = np.array([[((x-nmo_coord[2]/2)**2+(y-nmo_coord[3]/2)**2)**gamma/((nmo_coord[2]/2)**2+(nmo_coord[3]/2)**2)**gamma for x in range(nmo_coord[2])] for y in range(nmo_coord[3])])
            nom[nmo_coord[1]:nmo_coord[1]+nmo_coord[3], nmo_coord[0]:nmo_coord[0]+nmo_coord[2]] = \
            nom[nmo_coord[1]:nmo_coord[1]+nmo_coord[3], nmo_coord[0]:nmo_coord[0]+nmo_coord[2]]*mask[:,:,None] + abn[abn_coord[1]:abn_coord[1]+abn_coord[3], abn_coord[0]:abn_coord[0]+abn_coord[2]]*(1-mask[:,:,None])

            # add to occupied
            occupied.append((nmo_coord[0], nmo_coord[1], nmo_coord[0]+nmo_coord[2], nmo_coord[1]+nmo_coord[3]))

        if len(occupied) == 0:
            continue
        
        abnormal_count += 1
        cv2.imwrite(gen_path, nom)
        cv2.imwrite(nom_path, nom_copy)

        # write label.txt
        with open(label_path, 'w+') as f:
            for (xmin, ymin, xmax, ymax) in occupied:
                x = (xmin + xmax) / 2 / iw
                y = (ymin + ymax) / 2 / ih
                w = (xmax - xmin) / iw
                h = (ymax - ymin) / ih
                f.write('{} {} {} {} {}\n'.format(0, x, y, w, h))
        print(abnormal_count)
    add_paths = glob.glob(out_dir+'/*.jpg')
    with open(os.path.join(out_dir, 'train.txt'), 'w+') as f:
        f.writelines([_+'\n' for _ in add_paths])

generate_abnormal('dataset/neu/nom-scaled', 'dataset/neu/abn', 'dataset/neu/gen-scaled', thresh=2, gamma=1.5)

def append_train_txt(train_txt, abnormal_dir, out_txt):
    parent = str(Path(train_txt).parent)
    with open(train_txt, 'r+') as f:
        orig_paths = f.readlines()
    orig_paths = [_.replace('./', parent+'/') for _ in orig_paths]
    add_paths = glob.glob(abnormal_dir+'/*.jpg')
    with open(out_txt, 'w+') as f:
        f.writelines(orig_paths)
        f.writelines([_+'\n' for _ in add_paths])

append_train_txt('dataset/neu/IMAGES/train.txt', 'dataset/neu/gen-scaled', 'dataset/neu/train-aug-scaled.txt')