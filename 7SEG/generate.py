# sudo apt install chromium-browser
# pip install --upgrade html2image pandas

import os
import os.path as osp
import random
import tqdm
from html2image import Html2Image
import json
import numpy as np
import cv2
import concurrent.futures
import pathlib
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools

rand = random.Random(42)


def save_data(
    file_name: str, val: str, rand: random.Random, out_bbox: dict, base_id: int
) -> dict:
    margin = 2
    x1, y1 = (
        rand.randint(out_bbox["x1"], out_bbox["x1"] + margin),
        rand.randint(out_bbox["y1"], out_bbox["y1"] + margin),
    )
    x2, y2 = (
        rand.randint(out_bbox["x2"], out_bbox["x2"] + margin),
        rand.randint(out_bbox["y1"], out_bbox["y1"] + margin),
    )
    x3, y3 = (
        rand.randint(out_bbox["x2"], out_bbox["x2"] + margin),
        rand.randint(out_bbox["y2"], out_bbox["y2"] + margin),
    )
    x4, y4 = (
        rand.randint(out_bbox["x1"], out_bbox["x1"] + margin),
        rand.randint(out_bbox["y2"], out_bbox["y2"] + margin),
    )

    segm = np.array([x1, y1, x2, y2, x3, y3, x4, y4, x1, y1])

    cnt = np.array(segm).reshape(-1, 1, 2)
    x, y, w, h = cv2.boundingRect(cnt)
    ann = {
        "image_name": file_name,
        "segs": cnt.ravel().tolist(),
        "bbox": [x, y, w, h],
        "text": val,
        "base_id": base_id,
    }
    with open(
        osp.join(
            osp.dirname(file_name),
            osp.splitext(osp.basename(file_name))[0] + ".json",
        ),
        "w",
    ) as fd:
        json.dump(ann, fd)


def replace_var(data, var_name: str, var_val: str):
    s_str = f"--{var_name}:"
    sidx = data.find(s_str)
    if sidx == -1:
        return data

    eidx = data.find(";", sidx + len(s_str))

    if eidx == -1:
        return data

    return data[: sidx + len(s_str)] + str(var_val) + data[eidx:]


def replace_let(data, let_name: str, let_val: str):
    s_str = f"let {let_name} ="
    sidx = data.find(s_str)
    if sidx == -1:
        return data

    eidx = data.find(";", sidx + len(s_str))
    if eidx == -1:
        return data

    return data[: sidx + len(s_str)] + str(let_val) + data[eidx:]


def get_scale(data):
    s_str = "scale:"
    sidx = data.find(s_str)
    if sidx == -1:
        return 1

    eidx = data.find(";", sidx + len(s_str))
    if eidx == -1:
        return 1

    return int(data[sidx + len(s_str) : eidx])


def get_out_bbox(data):
    s_str = "out_bbox:"
    sidx = data.find(s_str)
    if sidx == -1:
        raise TypeError("out_bbox not setup!")

    eidx = data.find(";", sidx + len(s_str))
    if eidx == -1:
        raise TypeError("out_bbox not setup!")

    return json.loads(data[sidx + len(s_str) : eidx])


def generate(i, output_path):
    base_id = rand.randint(1, 5)

    with open(f"./base/base_{base_id}.html") as fd:
        index_data = fd.read()

    new_index_data = str(index_data)

    new_index_data = replace_var(
        new_index_data, "base_color", str(rand.randint(0, 360))
    )
    new_index_data = replace_var(
        new_index_data, "on_light", str(rand.randint(50, 85)) + "%"
    )
    new_index_data = replace_var(
        new_index_data, "off_light", str(rand.randint(20, 35)) + "%"
    )
    new_index_data = replace_var(
        new_index_data, "bg_light", str(rand.randint(10, 17)) + "%"
    )
    new_index_data = replace_var(
        new_index_data, "digit_skew", str(rand.randint(-15, 15)) + "deg"
    )

    new_index_data = replace_var(new_index_data, "bg_color", str(rand.randint(0, 360)))
    new_index_data = replace_var(
        new_index_data, "background_opacity", str(rand.randint(0, 35) / 100)
    )

    new_val = str(i)
    add_dot = rand.random() > 0.5
    add_minus = rand.random() > 0.5
    max_digits = rand.randint(3, 6)
    if add_minus:
        new_val = str(-int(new_val))

    if add_dot:
        new_val = new_val.strip()
        dot_idx = rand.randint(0, len(new_val))
        new_val = (new_val[:dot_idx] + "." + new_val[dot_idx:]).rjust(5)

    new_val = new_val.strip()
    if len(new_val.replace(".", "")) < max_digits:
        new_val = new_val.rjust(max_digits)

    new_index_data = replace_let(new_index_data, "value", f'"{new_val}"')

    new_index_data = replace_let(
        new_index_data, "digits", len(new_val) - 1 if add_dot else len(new_val)
    )

    new_index_data = replace_let(
        new_index_data, "thickness", float(rand.randint(6, 10) / 10.0)
    )

    fn = f"./base/index_tmp_{i}.html"
    out_fn = f"img_{base_id}_{i}_{add_minus}_{add_dot}_{max_digits}.jpg"

    with open(fn, "w") as fd:
        fd.write(new_index_data)

    scale = get_scale(new_index_data)
    size = (285 * scale, 200 * scale)

    hti = Html2Image(
        output_path=output_path, disable_logging=True, custom_flags=["--headless"]
    )
    hti.screenshot(url=fn, size=size, save_as=out_fn)
    os.remove(fn)

    if osp.exists(os.path.join(output_path, out_fn)):
        save_data(
            os.path.join(output_path, out_fn),
            new_val.strip(),
            rand=rand,
            out_bbox=get_out_bbox(new_index_data),
            base_id=base_id,
        )


def remove_all(folder):
    os.system(f"rm -rf {folder}")


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

    cv2.imwrite(f".examples/base_{row['base_id']}.jpg", image)


def run(f, iterator, images_dir):
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            tqdm.tqdm(
                executor.map(f, iterator, itertools.repeat(images_dir)),
                total=len(iterator),
            )
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="./7_seg_dataset")
    parser.add_argument("--display_first", type=bool, default=True)

    args = parser.parse_args()

    remove_all(args.images_dir)
    os.makedirs(args.images_dir)

    my_iter = range(1, 9999, 3)
    run(generate, my_iter, args.images_dir)
    # generate(9999, args.images_dir)

    out_dataset = defaultdict(list)

    for file in pathlib.Path(args.images_dir).glob("*.json"):
        with open(file) as fd:
            row = json.load(fd)

        out_dataset[row["image_name"]].append(
            {
                "text": row["text"],
                "bbox": row["bbox"],
                "segs": row["segs"],
                "base_id": row["base_id"],
            }
        )

    with open(os.path.join(args.images_dir, "ann_file.jsonl"), "w") as fd:
        fd.write("")

    disp_base = defaultdict(bool)
    for image_file_name, texts in out_dataset.items():
        if args.display_first:
            if not disp_base[texts[0]["base_id"]]:
                display_data(texts, image_file_name)
                disp_base[texts[0]["base_id"]] = True

            args.display_first = all(list(disp_base.values()))

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
