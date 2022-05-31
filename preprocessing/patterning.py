import numpy as np
import cv2
import os
pattern = []
pattern.append(np.zeros((4, 4)))
pattern.append(np.array([[1,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]))
pattern.append(np.array([[1,0,0,0], [0,0,0,0], [0,0,1,0], [0,0,0,0]]))
pattern.append(np.array([[1,0,1,0], [0,0,0,0], [0,0,1,0], [0,0,0,0]]))
pattern.append(np.array([[1,0,1,0], [0,0,0,0], [1,0,1,0], [0,0,0,0]]))
pattern.append(np.array([[1,0,1,0], [1,0,0,0], [1,0,1,0], [0,0,0,0]]))
pattern.append(np.array([[1,0,1,0], [1,0,0,0], [1,0,1,0], [0,0,1,0]]))
pattern.append(np.array([[1,0,1,0], [1,0,1,0], [1,0,1,0], [0,0,1,0]]))
pattern.append(np.array([[1,0,1,0], [1,0,1,0], [1,0,1,0], [1,0,1,0]]))
pattern.append(np.array([[1,1,1,0], [1,0,1,0], [1,0,1,0], [1,0,1,0]]))
pattern.append(np.array([[1,1,1,0], [1,0,1,0], [1,0,1,1], [1,0,1,0]]))
pattern.append(np.array([[1,1,1,1], [1,0,1,0], [1,0,1,1], [1,0,1,0]]))
pattern.append(np.array([[1,1,1,1], [1,0,1,0], [1,1,1,1], [1,0,1,0]]))
pattern.append(np.array([[1,1,1,1], [1,1,1,0], [1,1,1,1], [1,0,1,1]]))
pattern.append(np.array([[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,0,1,1]]))
pattern.append(np.ones((4, 4)))
#for i in pattern:
#    print(i)
dir_name = ['./adv_imgs_resnet_fgsm', './adv_imgs_resnet_ifgsm', './adv_imgs_vgg16_fgsm', 'adv_imgs_vgg16_ifgsm']
out_name = ['./CIFAR_resnet_fgsm', './CIFAR_resnet_ifgsm', './CIFAR_vgg16_fgsm', './CIFAR_vgg16_ifgsm']
for dname, oname in zip(dir_name, out_name):
    if not os.path.exists(oname):
        os.makedirs(oname)
    img_list = os.listdir(dname)
    for list in img_list:
        print(list)
        img = cv2.imread(dname + '/' + list)
        print(img.shape)
        res = np.zeros((img.shape[0] * 4, img.shape[1] * 4, img.shape[2]))
        for i in range(img.shape[2]):
            for j in range(img.shape[0]):
                for k in range(img.shape[1]):
                    level = img[j][k][i] // 16
                    for jj in range(4*j, 4*j + 4):
                        for kk in range(4*k, 4*k + 4):
                            res[jj][kk][i] = pattern[level][jj % 4][kk % 4] * 255
        img = cv2.imwrite(oname + '/' + list, res)
