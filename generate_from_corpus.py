import argparse
import os
import random
from copy import deepcopy

import cv2
import pandas as pd

from handwriting_generator import HandwritingGenerator
from transforms import get_cyrillic_transforms, get_hkr_transforms
from utils import font2not_allowed_symbols, default_config, allowed_symbols
from multiprocessing import Pool, Lock, Value


dataset2symbols = {
    "hkr": set(' !(),-.:;?HoАБВГДЕЖЗИЙКЛМНОПРСТУФХЧШЩЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёғҚқҮӨө–—…'),
    "cyrillic": set(' !"%\'()+,-./0123456789:;=?R[]abcehinoprstuxy«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№')
}

allowed_symbols_set = set(allowed_symbols)

transforms_dict = {
    "hkr": get_hkr_transforms(0.5),
    "cyrillic": get_cyrillic_transforms(0.5)
}

fonts_dir = "fonts"
img_dir = "img"


def process_word(i: int, word: str) -> dict:
    default_dict = dict(path=None, word=None)
    word = word.strip()
    if len(word) == 0:
        return default_dict

    word_set = set(word)
    if len(word_set.intersection(allowed_symbols_set)) != len(word_set):
        return default_dict

    font_name = None
    while font_name is None:
        font_name = random.choice(available_fonts)
        not_allowed_set = set(font2not_allowed_symbols.get(font_name, ""))
        if len(word_set.intersection(not_allowed_set)) != 0:
            font_name = None

    config["font_name"] = os.path.join(fonts_dir, font_name)
    config["font_size"] = random.randint(20, 100)
    img_generator.change_config(new_config=config)

    word_img = img_generator.write_word(word)
    if word_img is None:
        return default_dict

    img_name = f'synthetic_{i:06d}.png'
    try:
        cv2.imwrite(os.path.join(args.out_dir, img_dir, img_name), word_img)
    except Exception as e:
        return default_dict
    data_dict = dict(path=f"{img_dir}/{img_name}", word=word)
    return data_dict


mutex = Lock()
n_processed = Value('i', 0)


def func_wrapper(item):
    res = process_word(item[0], item[1])
    with mutex:
        global n_processed
        n_processed.value += 1
        if n_processed.value % 10 == 0:
            print(f"\r{n_processed.value} objects are processed...", end='', flush=True)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--corpus_path", required=True, type=str)
    parser.add_argument("--dataset_name", default="", type=str)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, img_dir), exist_ok=True)
    config = deepcopy(default_config)
    transforms_list = transforms_dict.get(args.dataset_name)
    symbols = dataset2symbols.get(args.dataset_name, allowed_symbols_set)
    allowed_symbols_set = allowed_symbols_set.intersection(symbols)

    img_generator = HandwritingGenerator(config=config, transforms=transforms_list)
    with open(args.corpus_path, "r") as f:
        words = f.readlines()

    available_fonts = [font_name for font_name in os.listdir(fonts_dir) if font_name.lower().endswith((".ttf", ".otf"))]

    with Pool(processes=8) as pool:
        res = pool.map(func_wrapper, enumerate(words))

    df = pd.DataFrame(res)
    df = df.dropna(axis=0)
    df.to_csv(os.path.join(args.out_dir, "gt.txt"), sep=",", index=False, header=False)
    print(f"\n{df.shape[0]} images created")
