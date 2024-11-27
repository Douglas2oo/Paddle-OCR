
from paddleocr import PaddleOCR,draw_ocr
import os

# ocr = PaddleOCR(lang='ch') # need to run only once to download and load model into memory
model_path = 'E:\DC_Project\Essay'
ocr = PaddleOCR(use_angle_cls=True, lang='ch', det_model_dir=model_path + '\ch_PP-OCRv4_det_infer\inference.pdmodel', rec_model_dir=model_path + '\ch_PP-OCRv4_rec_infer\inference.pdmodel', cls_model_dir=model_path + '\ch_PP-OCRv4_cls_infer\inference.pdmodel')
img_path = 'b0.jpg'
result = ocr.ocr(img_path, cls=False)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('resultb0.jpg')

