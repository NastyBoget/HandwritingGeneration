import os
from copy import deepcopy
from typing import Tuple

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

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

    def write_word(self, text: str, margin: tuple = (10, 10)) -> np.ndarray:
        img_size = (self.config["font_size"] * 10, self.config["font_size"] * len(text) * 10, 3)
        img = np.zeros(img_size, dtype=np.uint8)
        img[:, :] = self.config["page_color"]
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        x, y = self.config["font_size"], self.config["font_size"]
        x_min, y_min, x_max, y_max = draw.textbbox((x, y), text, self.font)

        draw.text((x, y), text, self.config["text_color"], font=self.font)
        return np.array(img.crop((x_min - margin[0], y_min - margin[1], x_max + margin[0], y_max + margin[1])))


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
