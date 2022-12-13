import os
from copy import deepcopy

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
                self.x = coords.max(axis=1).max() + self.config["word_space"] * self.config["font_size"]

            self.y += self.config["font_size"] + self.config["line_gap"]
            self.x = self.config["margin_left"]


if __name__ == "__main__":
    config = deepcopy(default_config)
    test_text = "АБВГДЕЖЗИЙКЛМН ОПРСТУФХЦЧШЩЬЫЪЭЮЯ абвгдежзийклмн опрстуфхцчшщьыъэюя. ! \"  %  (  ) , - ?  :  ;"
    font_dir = "fonts"
    text_generator = HandwritingGenerator(config=config)

    for font_name in sorted(os.listdir("fonts")):
        if not font_name.lower().endswith((".ttf", ".otf")):
            continue

        config["font_name"] = os.path.join(font_dir, font_name)
        text_generator.change_config(new_config=config)
        text_generator.write_text(f"{font_name[:-4]}: {test_text}")

    for j, img in enumerate(text_generator.pages + [text_generator.img]):
        cv2.imwrite(os.path.join(f"images", f'{j}.png'), img)
