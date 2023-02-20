import os
import random
from copy import deepcopy
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

from transforms import get_transforms

default_config = dict(
    font_name='fonts/Example.ttf',
    # Font size in px
    font_size=100,
    # Page size in px
    page_size=(2480, 3508),
    # Page color in RGB
    page_color=(255, 255, 255),
    # Text color in RGB
    text_color=(0, 0, 0),
    # Space between words
    word_space=1,
    # Vertical line gap
    line_gap=50,
    # Margins
    margin_left=100,
    margin_right=100,
    margin_top=100,
    margin_bottom=100
)


class HandwritingGenerator:

    def __init__(self, config: dict) -> None:
        self.config = config
        for key in default_config:
            if key not in self.config:
                self.config[key] = default_config[key]

        assert "font_name" in self.config
        self.font = ImageFont.truetype(self.config["font_name"], self.config["font_size"])
        self.x, self.y = self.config["margin_left"], self.config["margin_top"]
        self.img, self.draw = self.create_page()
        self.pages = []
        self.transforms = get_transforms(prob=0.5)

    def change_config(self, new_config: dict) -> None:
        for key in new_config:
            self.config[key] = new_config[key]
        self.font = ImageFont.truetype(self.config["font_name"], self.config["font_size"])

    def create_page(self) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        page_size = self.config["page_size"]
        img_arr = np.zeros((page_size[1], page_size[0], 3), dtype=np.uint8)
        img_arr[:, :] = self.config["page_color"]
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        return img, draw

    def delete_state(self) -> None:
        self.x, self.y = self.config["margin_left"], self.config["margin_top"]
        self.img, self.draw = self.create_page()
        self.pages = []

    def write_text(self, text: str) -> None:
        lines = text.split("\n")
        for line in lines:
            words = line.split()

            for word in words:
                x_min, y_min, x_max, y_max = self.draw.textbbox((self.x, self.y), word, self.font)
                if x_max + self.config["margin_right"] >= self.config["page_size"][0]:
                    self.x = self.config["margin_left"]
                    self.y += self.config["font_size"] + self.config["line_gap"]

                if y_max + self.config["margin_bottom"] >= self.config["page_size"][1]:
                    self.pages.append(self.img)
                    self.img, self.draw = self.create_page()
                    self.x, self.y = self.config["margin_left"], self.config["margin_top"]

                self.draw.text((self.x, self.y), word, self.config["text_color"], font=self.font)
                self.x = x_max + self.config["word_space"] * self.config["font_size"]

            self.y += self.config["font_size"] + self.config["line_gap"]
            self.x = self.config["margin_left"]

    def write_word(self, text: str, margin: tuple = (10, 10), randomize: bool = True) -> np.ndarray:
        self.config["font_size"] = random.randint(30, 100)
        img_size = (self.config["font_size"] * 10, self.config["font_size"] * len(text) * 10, 3)
        img = np.zeros(img_size, dtype=np.uint8)
        img[:, :] = self.config["page_color"]
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        x, y = self.config["font_size"], self.config["font_size"]
        x_min, y_min, x_max, y_max = draw.textbbox((x, y), text, self.font)

        draw.text((x, y), text, self.config["text_color"], font=self.font)
        img = np.array(img.crop((x_min - margin[0], y_min - margin[1], x_max + margin[0], y_max + margin[1])))  # noqa

        if randomize:
            img = self.skew(img, random.uniform(-1.2, 1.2))
            img = img.astype(np.uint8)
            for transform in self.transforms:
                img = transform(img)

        return img

    def skew(self, img: np.ndarray, angle: float) -> np.ndarray:
        """
        img should have 3 dimensions, background value 255 and black value 0
        :param angle: angle to rotate in radians [-1.5; 1.5]
        :param img: image to change
        :return: changed image
        """
        img = 255 - img
        if angle > 0:
            img2 = np.concatenate([img, np.zeros([img.shape[0], int(img.shape[0] * angle), 3])], axis=1)
        else:
            img2 = np.concatenate([np.zeros([img.shape[0], int(img.shape[0] * (-angle)), 3]), img], axis=1)

        M = np.float32([[1, -angle, 0], [0, 1, 0]])
        out_img = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        out_img = 255 - out_img
        return out_img


if __name__ == "__main__":
    config = deepcopy(default_config)
    test_text = "АБВГДЕЖЗИЙКЛМН ОПРСТУФХЦЧШЩЬЫЪЭЮЯ абвгдежзийклмн опрстуфхцчшщьыъэюя 0123456789 .!\"%(),-?:;"
    font_dir = "fonts"
    text_generator = HandwritingGenerator(config=config)

    for font_name in tqdm(sorted(os.listdir(font_dir))):
        if not font_name.lower().endswith((".ttf", ".otf")):
            continue
        config["font_name"] = os.path.join("fonts", font_name)
        text_generator.change_config(new_config=config)
        try:
            text_generator.write_text(test_text)
        except Exception as e:
            print(e)
        for j, img in enumerate(text_generator.pages + [text_generator.img]):
            img.save(os.path.join(f"images", f'{font_name}_{j}.png'))
        text_generator.delete_state()
