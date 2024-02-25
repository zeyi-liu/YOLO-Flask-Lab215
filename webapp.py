import io
import cv2
import numpy as np
from PIL import Image

import torch
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.plots import Annotator, colors

from flask import Flask, request, redirect, jsonify
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        result = {}
        if img is not None:
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=imgsz)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.half() if model.half() else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = model(img)[0]

                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres)
                
                annotator = Annotator(showimg, line_width=3, example=str(names))  #绘图
                # # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], showimg.shape).round()
                        # Write results
                        id = 0
                        for *xyxy, conf, cls in reversed(det):
                            id += 1
                            c = int(cls)
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            result[id] = {"xyxy":str([int(item.cpu()) for item in xyxy]), "label":names[c], "conf":str(float(conf.cpu()))}
                            print("xyxy:",[int(item.cpu()) for item in xyxy], "label:", label)
                            
                showimg = annotator.result()
                imgFile = "static/img.jpg"
                cv2.imwrite(imgFile, showimg)
    
    return jsonify(result)

    # return render_template("index.html")


if __name__ == "__main__":
    weights = 'best.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = attempt_load(weights, map_location=device)
    model.to(device).eval()
    model.half()
    conf_thres = 0.25  # NMS置信度
    iou_thres = 0.45  # IOU阈值
    # 从模型中获取各类别名称
    names = model.module.names if hasattr(model, 'module') else model.names
    # 给每一个类别初始化颜色
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    stride = int(model.stride.max())
    imgsz = check_img_size(640, s=stride)

    app.run(host="0.0.0.0", port=5000)  # debug=True causes Restarting with stat
