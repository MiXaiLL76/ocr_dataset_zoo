import xml.etree.ElementTree as ET
import os
import cv2
import json
import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np


def process_xml_file(file, labels_dir, images_dir):
    tree = ET.parse(f"{labels_dir}/{file}")
    root = tree.getroot()
    image = root.find("image")
    image_file_name = os.path.join(images_dir, image.get("file"))

    texts = []
    for box in image.findall("box"):
        x, y, w, h = (
            int(box.get("left")),
            int(box.get("top")),
            int(box.get("width")),
            int(box.get("height")),
        )
        label = box.find("label").text
        segs = [float(c) for c in box.find("segs").text.split(",")]
        texts.append(
            {
                "text": label,
                "bbox": [x, y, w, h],
                "segs": segs,
            }
        )
    return image_file_name, texts


def process_txt_file(file, labels_dir, images_dir):
    texts = []
    image_id = int(os.path.basename(file).replace(".txt", ""))
    image_file_name = os.path.join(images_dir, f"{image_id}.jpg")

    with open(f"{labels_dir}/{file}", "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if line == "":
            continue

        segs, label = line.split("####")
        _segs = np.array(
            [int(c) for c in segs.split(",") if c != ""], dtype=np.int32
        ).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(_segs)
        texts.append(
            {
                "text": label,
                "bbox": [x, y, w, h],
                "segs": _segs.flatten().tolist(),
            }
        )

    return image_file_name, texts


def process_file_dir(labels_dir, images_dir):
    for file in tqdm.tqdm(os.listdir(labels_dir)):
        if file.endswith(".xml"):
            image_file_name, texts = process_xml_file(file, labels_dir, images_dir)
        elif file.endswith(".txt"):
            image_file_name, texts = process_txt_file(file, labels_dir, images_dir)
        else:
            raise ValueError(f"Unknown file type: {file}")

        yield image_file_name, texts


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
    parser.add_argument("--labels_dir", type=str, default="./ctw1500_train_labels")
    parser.add_argument("--images_dir", type=str, default="./train_images")
    parser.add_argument("--display_first", type=bool, default=True)

    args = parser.parse_args()

    os.makedirs(args.images_dir, exist_ok=True)

    with open(os.path.join(args.images_dir, "ann_file.jsonl"), "w") as fd:
        fd.write("")

    for image_file_name, texts in process_file_dir(args.labels_dir, args.images_dir):
        # image = cv2.imread(image_file_name)
        # height, width, _ = image.shape
        if args.display_first:
            display_data(texts, image_file_name)
            args.display_first = False

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
