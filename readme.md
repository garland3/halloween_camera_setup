# halloween_camera_setup 

Trying to use some ML to process my security camera and scare people at the door. 


1. use usb camera to get images
   1. I hooked this up to my home security camera. Making them a webcam basically. 
      1. [hdmi splitter](https://www.amazon.com/gp/product/B005HXFARS/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1) so that one signal goes to my screen and one can go to my computer as a webcam. 
      2. [hdmi to usb](https://www.amazon.com/gp/product/B09FLN63B3/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1)
   2. basically I put a hdmi fork on the hdmi output, and then a hdmi to usb dongle
2. using yolo
   1.    # https://github.com/ultralytics/yolov5
3. for audio
   1. I got audio from a youtube video. see [src\utilslocal\audiotest.py](src\utilslocal\audiotest.py)
   2. I'm trying to make it talk to the people. 





# OLD
1. use yolo
2. 
   1. using hf
      1. https://huggingface.co/docs/transformers/model_doc/yolos
      2. https://huggingface.co/hustvl/yolos-small
      3. https://huggingface.co/spaces/Gradio-Blocks/Object-Detection-With-DETR-and-YOLOS/blob/main/app.py
      4. 
   2. test yolo output using this guy. 
      1. https://huggingface.co/spaces/oussamamatar/yolo-mediapipe/blob/main/app.py