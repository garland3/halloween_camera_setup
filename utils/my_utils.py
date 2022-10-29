# from flask import Flask, render_template, Response, make_response
from typing import List
import cv2
import base64
from pathlib import Path
import argparse
import re

def get_img_dir()->Path:
    outputdir = Path("imgs")
    outputdir.mkdir(exist_ok=True)
    return outputdir

def get_imgs_in_dir() -> List[Path]:
    dir = get_img_dir()
    imgs = list(dir.glob("*.png"))
    return imgs

def get_highest_img_save_index()->int:
    imgs =get_imgs_in_dir()
    numbers  = [re.findall(r"\d+",str(a.name))[0] for a in imgs]
    # print()
    for n,i in zip(numbers, imgs):
        print(n,i.name)
    numbers_ints = [int(a) for a in numbers]
    if len(numbers_ints)>0:
        mymax =  max(numbers_ints)
    else:
        mymax = 0
    print(f"The max number is {mymax}")
    return mymax

def camera_loop():
    outputdir = get_img_dir()
    print("starting camera")
    camera = cv2.VideoCapture(1)
    print("camera started")
    
    cnt = 0
    while True:
        print("Loop started. ")
        
        ret, frame = camera.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == ord('q'):
            break
        cnt+=1
        img_name = outputdir / f"img{cnt}.png"
        cv2.imwrite(str(img_name), frame)
        
    camera.release()
    cv2.destroyAllWindows()

def parse_inputs():
    parser  = argparse.ArgumentParser('security camera capture and ML')
    parser.add_argument("--usbcam", action = 'store_true', default=False)
    args = parser.parse_args()
    return args


# if __name__ == "__main__":
#     args = parse_inputs()
#     print(args)
#     if args.usbcam:
#         camera_loop()
#     else:
#         get_highest_img_save_index()
        

