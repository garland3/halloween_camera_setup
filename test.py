import argparse

from utils.my_utils import camera_loop, get_highest_img_save_index


def parse_inputs():
    parser  = argparse.ArgumentParser('security camera capture and ML')
    parser.add_argument("--usbcam", action = 'store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_inputs()
    print(args)
    if args.usbcam:
        camera_loop()
    else:
        get_highest_img_save_index()
        

