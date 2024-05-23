from detection import detect_one_img

import os




def detect_imgs(dir_path):
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        res = detect_one_img(file_path)
        print(file_path, res)
    

if __name__ == '__main__':
    dir_path = 'data/Drink_284_Detection_coco/test'
    detect_imgs(dir_path)

