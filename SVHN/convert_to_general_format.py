import os
import cv2
import json
import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
import mat73
from collections import defaultdict

def process_svhn(images_dir):
    mat_file = mat73.loadmat(os.path.join(images_dir, "digitStruct.mat"))
    
    dSName = mat_file['digitStruct']['name']
    dSBbox = mat_file['digitStruct']['bbox']

    for i in tqdm.tqdm(range(len(dSName))):
        bbox_dict = dSBbox[i]
        image_name = os.path.basename(dSName[i])

        x1 = int(np.array(bbox_dict['left']).min())
        y1 = int(np.array(bbox_dict['top']).min())
        x2 = int((np.array(bbox_dict['left']) + np.array(bbox_dict['width'])).max())
        y2 = int((np.array(bbox_dict['top']) + np.array(bbox_dict['height'])).max())
        
        if bbox_dict['label'] is None:
            continue
        
        try:
            # 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10. 
            text = ""
            for label in bbox_dict['label']:
                if int(label) == 10:
                    label = "0"
                text += str(int(label))
        except TypeError:
            continue

        row = {
            "image_name" : os.path.join(images_dir, image_name),
            "text": text,
            "bbox": [x1, y1, x2-x1, y2-y1]
        }

        yield row
    

def display_data(row):
    data = cv2.imread(row['image_name'])

    fig, ax = plt.subplots()

    title = []
    title.append(f"{row['text']}")
    cv2.rectangle(
        data,
        (int(row["bbox"][0]), int(row["bbox"][1])),
        (
            int(row["bbox"][0]) + int(row["bbox"][2]),
            int(row["bbox"][1]) + int(row["bbox"][3]),
        ),
        (0, 255, 0),
        2,
    )
    title_kargs = {}
    ax.set_title(" | ".join(title), **title_kargs)
    ax.imshow(data)
    fig.canvas.draw()  # Draw the canvas, cache the renderer

    # Convert the canvas to a raw RGB buffer
    buf = fig.canvas.buffer_rgba()
    ncols, nrows = fig.canvas.get_width_height()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    key = "_".join(title)
    cv2.imwrite(f".examples/{key}_{os.path.basename(row['image_name'])}", image)

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="./train_images")
    parser.add_argument("--display_first", type=bool, default=True)

    args = parser.parse_args()

    os.makedirs(args.images_dir, exist_ok=True)

    with open(os.path.join(args.images_dir, "ann_file.jsonl"), "w") as fd:
        fd.write("")
    
    out_dataset = defaultdict(list)

    for row in process_svhn(args.images_dir):
        if args.display_first:
            display_data(row)
            args.display_first = False

        out_dataset[row['image_name']].append({
            "text" : row['text'],
            "bbox" : row['bbox']
        })

    with open(os.path.join(args.images_dir, "ann_file.jsonl"), "w") as fd:
        fd.write("")
    
    for image_file_name, texts in out_dataset.items():
        with open(os.path.join(args.images_dir, "ann_file.jsonl"), "a") as fd:
            fd.write(
                json.dumps(
                    {
                        "filename": os.path.basename(image_file_name),
                        "annotations": texts,
                    }
                )
            )
            fd.write("\n")