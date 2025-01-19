import os
import cv2
import json
import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def process_file_dir(labels_file, images_dir):
    with open(labels_file) as fd:
        data = json.load(fd)

    for _, ann in tqdm.tqdm(data["anns"].items()):
        label = ann["utf8_string"]
        image = data["imgs"][ann["image_id"]]

        x, y, w, h = np.array(ann["bbox"]).astype(int).tolist()

        file_name = os.path.basename(image["file_name"])
        file_name = os.path.join(images_dir, file_name)

        if not os.path.exists(file_name):
            continue

        row = {
            "image_name": file_name,
            "text": label,
            "bbox": [x, y, w, h],
            "segs": ann["points"],
        }

        yield row


def display_data(texts, image_file_name):
    data = cv2.imread(image_file_name)

    fig, ax = plt.subplots()

    title = []
    for idx, row in enumerate(texts):
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
        cv2.drawContours(
            data,
            [np.array(row["segs"], dtype=np.int32).reshape(-1, 2)],
            -1,
            (255, 0, 0),
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
    cv2.imwrite(f".examples/{os.path.basename(image_file_name)}", image)

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="./TextOCR_0.1_train.json")
    parser.add_argument("--images_dir", type=str, default="./train_images")
    parser.add_argument("--display_first", type=bool, default=True)

    args = parser.parse_args()

    os.makedirs(args.images_dir, exist_ok=True)

    out_dataset = defaultdict(list)

    for row in process_file_dir(args.labels, args.images_dir):
        out_dataset[row["image_name"]].append(
            {"text": row["text"], "bbox": row["bbox"], "segs": row["segs"]}
        )

    ann_file = "train_ann_file.jsonl"
    if "val" in args.labels:
        ann_file = "val_ann_file.jsonl"

    with open(os.path.join(args.images_dir, ann_file), "w") as fd:
        fd.write("")

    for image_file_name, texts in out_dataset.items():
        if len(texts) < 5:
            if args.display_first:
                display_data(texts, image_file_name)
                args.display_first = False

        with open(os.path.join(args.images_dir, ann_file), "a") as fd:
            fd.write(
                json.dumps(
                    {
                        "filename": os.path.basename(image_file_name),
                        "annotations": texts,
                    }
                )
            )
            fd.write("\n")
