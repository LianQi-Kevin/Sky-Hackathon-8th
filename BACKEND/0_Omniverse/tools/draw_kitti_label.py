import glob
import os

from PIL import Image, ImageDraw
from tqdm import tqdm


def main(img_path: str, label_path: str, output_path: str, allow_label: list):
    os.makedirs(output_path, exist_ok=True)
    for img_path in tqdm(glob.glob(os.path.join(img_path, "*.png"))):
        id_, _ = os.path.splitext(os.path.basename(img_path))
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        with open(os.path.join(label_path, f"{id_}.txt")) as label_f:
            for line in label_f.readlines():
                line_list = line.split(" ")
                name, x1, y1, x2, y2 = line_list[0], line_list[4], line_list[5], line_list[6], line_list[7]
                if name in allow_label:
                    draw.rectangle(xy=(float(x1), float(y1), float(x2), float(y2)), outline="red", width=3)
        img.save(os.path.join(output_path, f"{id_}.png"))


if __name__ == '__main__':
    main(img_path="../_output/Camera/rgb",
         label_path="../_output/Camera/object_detection",
         output_path="_output",
         allow_label=["box"])
