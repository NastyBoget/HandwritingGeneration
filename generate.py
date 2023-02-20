import os
import random
from copy import deepcopy

import cv2
import pandas as pd
from tqdm import tqdm

from handwriting_generator import HandwritingGenerator, default_config
from text_generator import get_random_text

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

MAX_LEN = 3


if __name__ == "__main__":
    fonts_dir = "fonts"
    out_dir = "synthetic"
    img_dir = "img"
    os.makedirs(os.path.join(out_dir, img_dir), exist_ok=True)
    config = deepcopy(default_config)
    text_generator = HandwritingGenerator(config=config)
    data_dict = {"path": [], "word": []}

    for font_name in tqdm(sorted(os.listdir(fonts_dir))):
        if not font_name.lower().endswith((".ttf", ".otf")):
            continue

        config["font_name"] = os.path.join(fonts_dir, font_name)
        text_generator.change_config(new_config=config)

        sentences = set()
        while len(sentences) < 60000:
            sentences_list = []
            words = get_random_text().split(" ")

            # group words into sentences
            while len(words) > MAX_LEN:
                sentence_len = random.randint(1, MAX_LEN)
                sentence = " ".join(words[:sentence_len])
                words = words[sentence_len:]

                if font_name in font2not_allowed_symbols:
                    for sym in font2not_allowed_symbols[font_name]:
                        sentence = sentence.replace(sym, "")

                if not sentence:
                    continue
                sentences_list.append(sentence)

            sentences = sentences | set(sentences_list)

        sentences = list(sentences)[:60000]

        for i, sentence in enumerate(sentences):
            text_generator.config["font_size"] = random.randint(20, 100)
            sentence_img = text_generator.write_word(sentence)
            if sentence_img is not None:
                img_name = f'{font_name}_{i:04d}.png'
                try:
                    cv2.imwrite(os.path.join(out_dir, img_dir, img_name), sentence_img)
                except Exception as e:
                    continue
                data_dict["path"].append(f"{img_dir}/{img_name}")
                data_dict["word"].append(sentence)

        df = pd.DataFrame(data_dict)
        df.to_csv(os.path.join(out_dir, "gt.txt"), sep="\t", index=False, header=False)

    df = pd.DataFrame(data_dict)
    df.to_csv(os.path.join(out_dir, "gt.txt"), sep="\t", index=False, header=False)
