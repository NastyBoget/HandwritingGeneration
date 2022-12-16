import os
from copy import deepcopy
from typing import Optional

import cv2
import numpy as np


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
        self.font = cv2.freetype.createFreeType2()
        self.font.loadFontData(fontFileName=self.config["font_name"], id=0)
        self.x, self.y = self.config["margin_left"], self.config["margin_top"] + self.config["font_size"]
        self.img = self.create_page()
        self.pages = []

    def change_config(self, new_config: dict) -> None:
        for key in new_config:
            self.config[key] = new_config[key]
        self.font = cv2.freetype.createFreeType2()
        self.font.loadFontData(fontFileName=self.config["font_name"], id=0)

    def create_page(self) -> np.ndarray:
        page_size = self.config["page_size"]
        img = np.zeros((page_size[1], page_size[0], 3), dtype=np.uint8)
        img[:] = 255
        return img

    def delete_state(self) -> None:
        self.x, self.y = self.config["margin_left"], self.config["margin_top"] + self.config["font_size"]
        self.img = self.create_page()
        self.pages = []

    def write_text(self, text: str) -> None:
        lines = text.split("\n")
        for line in lines:
            words = line.split()

            for word in words:
                if self.x + self.config["font_size"] * len(word) + self.config["margin_right"] >= self.config["page_size"][0]:
                    self.x = self.config["margin_left"]
                    self.y += self.config["font_size"] + self.config["line_gap"]

                if self.y + self.config["margin_bottom"] >= self.config["page_size"][1]:
                    self.pages.append(self.img)
                    self.img = self.create_page()
                    self.x, self.y = self.config["margin_left"], self.config["margin_top"] + self.config["font_size"]

                try:
                    self.font.putText(img=self.img, text=word, org=(self.x, self.y), fontHeight=self.config["font_size"],
                                      color=(0, 0, 0), thickness=-1, line_type=cv2.LINE_AA, bottomLeftOrigin=True)
                except cv2.error as e:
                    print(e)

                cropped = self.img[self.y - int(self.config["font_size"] / 2):self.y]
                coords = np.argwhere(cropped == 0)
                if coords.size != 0:
                    self.x = coords.max(axis=0)[1] + self.config["word_space"] * self.config["font_size"]

            self.y += self.config["font_size"] + self.config["line_gap"]
            self.x = self.config["margin_left"]

    def write_word(self, text: str) -> Optional[np.ndarray]:
        img_size = (self.config["font_size"] * 10, self.config["font_size"] * len(text) * 10, 3)
        img = np.zeros(img_size, dtype=np.uint8)
        img[:] = 255
        x = self.config["font_size"] * 10
        y = self.config["font_size"] * 5
        self.font.putText(img=img, text=text, org=(x, y), fontHeight=self.config["font_size"],
                          color=(0, 0, 0), thickness=-1, line_type=cv2.LINE_AA, bottomLeftOrigin=True)
        coords = np.argwhere(img == 0)
        if coords.size == 0:
            return None

        y_min, x_min, _ = coords.min(axis=0)
        y_max, x_max, _ = coords.max(axis=0)
        img = img[y_min - int((y_max - y_min) / 5):y_max + int((y_max - y_min) / 5),
                  x_min - int(self.config["font_size"] / 10):x_max + int(self.config["font_size"] / 5)]
        return img


if __name__ == "__main__":
    config = deepcopy(default_config)
    test_text = "АБВГДЕЖЗИЙКЛМН ОПРСТУФХЦЧШЩЬЫЪЭЮЯ абвгдежзийклмн опрстуфхцчшщьыъэюя 0123456789 .!\"%(),-?:;"
    font_dir = "fonts"
    text_generator = HandwritingGenerator(config=config)

    for font_name in sorted(os.listdir(font_dir)):
        if not font_name.lower().endswith((".ttf", ".otf")):
            continue
        config["font_name"] = os.path.join("fonts", font_name)
        text_generator.change_config(new_config=config)
        try:
            text_generator.write_text(test_text)
        except Exception as e:
            print(e)
        for j, img in enumerate(text_generator.pages + [text_generator.img]):
            cv2.imwrite(os.path.join(f"images", f'{font_name}_{j}.png'), img)
        text_generator.delete_state()
