import os
import cv2
import json
import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def process_icdar(labels, images_dir):
    with open(labels) as fd:
        data = json.load(fd)

    data_list = data.get("data_list", [])

    for item in tqdm.tqdm(data_list):
        text = item["instances"][0]["text"]
        image_name = os.path.basename(item["img_path"])
        image_file_name = os.path.join(images_dir, image_name)
        img = cv2.imread(image_file_name, cv2.IMREAD_UNCHANGED)
        x1, y1 = 0, 0
        y2, x2 = (np.array(img.shape[:2]) - [1, 1]).tolist()

        row = {
            "image_name": image_file_name,
            "text": text,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
        }

        yield row


def display_data(row):
    data = cv2.imread(row["image_name"])

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
    parser.add_argument("--labels", type=str, default="./ic13_train_labels.json")
    parser.add_argument(
        "--images_dir", type=str, default="./ic13_textrecog_train_img_gt"
    )
    parser.add_argument("--display_first", type=bool, default=True)

    args = parser.parse_args()

    os.makedirs(args.images_dir, exist_ok=True)

    out_dataset = defaultdict(list)

    for row in process_icdar(args.labels, args.images_dir):
        if args.display_first:
            display_data(row)
            args.display_first = False

        out_dataset[row["image_name"]].append(
            {"text": row["text"], "bbox": row["bbox"]}
        )

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
