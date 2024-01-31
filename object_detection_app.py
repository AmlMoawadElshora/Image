from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

app = Flask(__name__)

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
model.eval()

def make_prediction(img): 
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
                                          colors=["red" if label=="person" else "green" for label in prediction["labels"]] , width=2)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1,2,0)
    return img_with_bboxes_np

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]
        img = Image.open(file)

        prediction = make_prediction(img)
        img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2,0,1), prediction)

        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111)
        plt.imshow(img_with_bbox)
        plt.xticks([],[])
        plt.yticks([],[])
        ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

        plt.savefig("static/img.png")

        del prediction["boxes"]
        return jsonify(prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
