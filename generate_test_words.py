import os
import random
from copy import deepcopy

import cv2
import pandas as pd
from tqdm import tqdm

from handwriting_generator import HandwritingGenerator, default_config

font2not_allowed_symbols = {
    "Abram.ttf": "Ъ",
    "Benvolio.ttf": "Ъ",
    "Capuletty.ttf": "Ъ",
    "Eskal.ttf": "Ъ",
    "Gregory.ttf": "Ъ",
    "Gogol.ttf": "\"%()?",
    "Lorenco.ttf": "Ъ",
    "Marutya.ttf": "ЬЫЪ",
    "Merkucio.ttf": "Ъ",
    "Montekky.ttf": "Ъ",
    "Pushkin.ttf": "%",
    "Tesla.otf": "ЙЩЬЫЪЭЮЯйщьыъэюя",
    "Tibalt.ttf": "Ъ",
    "Voronov.ttf": "ЬЫЪ"
}
allowed_symbolds = 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯабвгдежзийклмнопрстуфхцчшщьыъэюя0123456789.!"%(),\-?:; '

MAX_LEN = 3


if __name__ == "__main__":
    fonts_dir = "fonts"
    out_dir = "synthetic"
    img_dir = "img"
    os.makedirs(os.path.join(out_dir, img_dir), exist_ok=True)
    config = deepcopy(default_config)
    text_generator = HandwritingGenerator(config=config)
    df = pd.read_csv("test1_hkr_gt.txt", sep="\t", names=["path", "word"])
    data_dict = {"path": [], "word": []}
    words = df.word.unique()

    for font_name in tqdm(sorted(os.listdir(fonts_dir))):
        if not font_name.lower().endswith((".ttf", ".otf")):
            continue

        config["font_name"] = os.path.join(fonts_dir, font_name)
        text_generator.change_config(new_config=config)

        for i, word in enumerate(words):
            fixed_word = word
            for sym in word:
                if sym in font2not_allowed_symbols.get(font_name, "") or sym not in allowed_symbolds:
                    fixed_word = fixed_word.replace(sym, "")

            if not fixed_word:
                continue

            text_generator.config["font_size"] = random.randint(20, 100)
            word_img = text_generator.write_word(fixed_word)
            if word_img is not None:
                img_name = f'{font_name}_{i:04d}.png'
                try:
                    cv2.imwrite(os.path.join(out_dir, img_dir, img_name), word_img)
                except Exception as e:
                    continue
                data_dict["path"].append(f"{img_dir}/{img_name}")
                data_dict["word"].append(fixed_word)

        df = pd.DataFrame(data_dict)
        df.to_csv(os.path.join(out_dir, "gt.txt"), sep="\t", index=False, header=False)

    df = pd.DataFrame(data_dict)
    df.to_csv(os.path.join(out_dir, "gt.txt"), sep="\t", index=False, header=False)
