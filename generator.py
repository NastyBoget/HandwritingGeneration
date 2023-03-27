import os
import random
from copy import deepcopy

import cv2
import pandas as pd
from tqdm import tqdm

from handwriting_generator import HandwritingGenerator, default_config
from text_generator import TextGenerator
from utils import font2not_allowed_symbols


class Generator:

    def __init__(self, max_sentence_len: int, max_word_len: int) -> None:
        self.max_sentence_len = max_sentence_len
        self.max_word_len = max_word_len
        self.img_generator = HandwritingGenerator(config=default_config)
        self.text_generator = TextGenerator()

    def generate_data(self, img_number: int, fonts_dir: str, out_dir: str, img_dir: str) -> None:
        os.makedirs(os.path.join(out_dir, img_dir), exist_ok=True)
        config = deepcopy(default_config)

        data_dict = {"path": [], "word": []}

        for font_name in tqdm(sorted(os.listdir(fonts_dir))):
            if not font_name.lower().endswith((".ttf", ".otf")):
                continue

            config["font_name"] = os.path.join(fonts_dir, font_name)
            self.img_generator.change_config(new_config=config)

            sentences = set()
            while len(sentences) < img_number:
                sentences_list = []
                words = [word for word in self.text_generator.get_random_text().split(" ") if len(word) < self.max_word_len]

                # group words into sentences
                while len(words) > self.max_sentence_len:
                    sentence_len = random.randint(1, self.max_sentence_len)
                    sentence = " ".join(words[:sentence_len])
                    words = words[sentence_len:]

                    if font_name in font2not_allowed_symbols:
                        for sym in font2not_allowed_symbols[font_name]:
                            sentence = sentence.replace(sym, "")

                    if not sentence:
                        continue
                    sentences_list.append(sentence)

                sentences = sentences | set(sentences_list)

            sentences = list(sentences)[:img_number]

            for i, sentence in enumerate(sentences):
                self.img_generator.config["font_size"] = random.randint(20, 100)
                sentence_img = self.img_generator.write_word(sentence)
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


if __name__ == "__main__":
    gen = Generator(max_sentence_len=3, max_word_len=40)
    gen.generate_data(60000, "fonts", "synthetic", "img")
