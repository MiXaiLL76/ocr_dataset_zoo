import os
import cv2
import json
import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd


def process_icdar2015(images_dir):
    dataframe = []
    with open(os.path.join(images_dir, "gt.txt"), encoding="utf-8-sig") as fd:
        for line in fd.readlines():
            dataframe.append(
                {
                    "file": line[: line.index(",")].strip(),
                    "text": line[line.index(",") + 1 :].strip()[1:-1].strip(),
                }
            )

    dataframe = pd.DataFrame(dataframe)
    coords_dataframe = pd.read_csv(
        os.path.join(images_dir, "coords.txt"),
        names=["file", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"],
        dtype={"file": str},
    )
    full_dataframe = dataframe.set_index("file").join(
        coords_dataframe.set_index("file")
    )

    for image_name, row in tqdm.tqdm(
        full_dataframe.iterrows(), total=len(full_dataframe)
    ):
        text = row["text"]
        segs = [
            row["x1"],
            row["y1"],
            row["x2"],
            row["y2"],
            row["x3"],
            row["y3"],
            row["x4"],
            row["y4"],
        ]
        bbox = [
            min(segs[::2]),
            min(segs[1::2]),
            max(segs[::2]) - min(segs[::2]),
            max(segs[1::2]) - min(segs[1::2]),
        ]
        row = {
            "image_name": os.path.join(images_dir, image_name),
            "text": text,
            "bbox": bbox,
            "segs": segs,
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

    key = "_".join(title).replace(" ", "_")
    cv2.imwrite(f".examples/{key}_{os.path.basename(image_file_name)}", image)

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir", type=str, default="./ic15_textrecog_train_img_gt"
    )
    parser.add_argument("--display_first", type=bool, default=True)

    args = parser.parse_args()

    os.makedirs(args.images_dir, exist_ok=True)

    out_dataset = defaultdict(list)

    for row in process_icdar2015(args.images_dir):
        out_dataset[row["image_name"]].append(
            {"text": row["text"], "bbox": row["bbox"], "segs": row["segs"]}
        )

    with open(os.path.join(args.images_dir, "ann_file.jsonl"), "w") as fd:
        fd.write("")

    for image_file_name, texts in out_dataset.items():
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
