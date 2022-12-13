import random
from typing import List

import cv2
import os

import numpy as np
from PIL import Image
from sklearn import linear_model
from tqdm import tqdm


class TemplateDrawer:
    """
    fill template for https://www.calligraphr.com
    """
    def __init__(self, template_dir: str, base_dir: str, sm_dir: str, cap_dir: str, sym_dir: str) -> None:

        self.template_dir = template_dir
        self.base_dir = base_dir
        self.sm_dir = sm_dir
        self.cap_dir = cap_dir
        self.sym_dir = sym_dir

        self.init_x = 36 + 25
        self.init_y = 315  # baseline
        self.x_step = 200
        self.y_step = 260
        self.sym2num = {'!': 0, '"': 1, ',': 2, '.': 3, '%': 4, '?': 5, '(': 6, ')': 7, '-': 8, ':': 9, ';': 10}
        self.baseline_letters = ['д', 'з', 'р', 'у', 'ф', 'ц', 'щ']

    @staticmethod
    def detect_baseline(img: np.ndarray, threshold: int = 20) -> int:
        low = []
        for w in range(1, img.shape[1] - 1):
            if np.max(img[:, w]) <= threshold:
                continue
            for h in range(img.shape[0] - 5, 0, -1):
                if img[h, w] > threshold:
                    low += [[h, w]]
                    break
        points_lower = np.array(low)
        x = points_lower[:, 1].reshape(points_lower.shape[0], 1)
        y = points_lower[:, 0].reshape(points_lower.shape[0], 1)
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        model_ransac.fit(x, y)
        y_mean = model_ransac.predict(np.array([img.shape[1] / 2]).reshape(1, -1))
        return int(y_mean)

    def __find_baseline(self, img: np.ndarray, letter: str) -> int:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            if letter in ['д', 'з', 'у']:
                y = self.detect_baseline(255 - gray[:int(0.6 * img.shape[0]), :])
            elif letter in ['Ц', 'Щ', 'р', 'ф', 'ц', 'щ']:
                y = self.detect_baseline(255 - gray)
            elif letter in ['"', ',', '-', ';']:
                y = 90
            else:
                coords = np.argwhere(gray == 0)
                y = coords.max(axis=0).max()
        except ValueError:
            y = int(img.shape[0] / 2)
        return y

    def fill_page(self, template_name: str, img_letters: List[List[str]], font_num: int, example_num: int) -> np.ndarray:
        x, y = self.init_x, self.init_y
        num = f"{font_num:04d}"
        ext = f"{example_num:02d}.png"
        img_type = random.randint(1, 4)

        img = cv2.imread(os.path.join(self.template_dir, template_name))

        for line in img_letters:
            x = self.init_x

            for letter in line:
                try:
                    img_name = f"{num}_{letter}_{ext}"
                    letter_img = None

                    if not (letter.isalpha() or letter.isdigit()):
                        sym_type = self.sym2num[letter]
                        if os.path.isfile(os.path.join(self.base_dir, self.sym_dir, f"{img_type}_{sym_type}.png")):
                            letter_img = cv2.imread(os.path.join(self.base_dir, self.sym_dir, f"{img_type}_{sym_type}.png"))
                    elif letter.isupper():
                        if os.path.isfile(os.path.join(self.base_dir, self.cap_dir, letter, img_name)):
                            letter_img = cv2.imread(os.path.join(self.base_dir, self.cap_dir, letter, img_name))
                    else:
                        if os.path.isfile(os.path.join(self.base_dir, self.sm_dir, letter, img_name)):
                            letter_img = cv2.imread(os.path.join(self.base_dir, self.sm_dir, letter, img_name))

                    # TODO adjust size and baseline by hand
                    if letter_img is not None:
                        y_b = self.__find_baseline(letter_img, letter)
                        img2insert = np.where(letter_img == 0, letter_img, img[y - y_b:y - y_b + 145, x:x + 145])
                        img[y - y_b:y - y_b + 145, x:x + 145] = img2insert
                except Exception as e:
                    print(e)

                x += self.x_step
            y += self.y_step

        return img


if __name__ == "__main__":
    first_img_letters = [
        ['!', '"', '%', '(', ')', ','],
        ['-', '?', '0', '1', '2', '3'],
        ['4', '5', '6', '7', '8', '9', ':', ';'],
        ['.', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж'],
        ['З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О'],
        ['П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц'],
        ['Ч', 'Ш', 'Щ', 'Э', 'Ю', 'Я', 'а', 'б'],
        ['в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й'],
    ]
    second_img_letters = [
        ['к', 'л', 'м', 'н', 'о', 'п'],
        ['р', 'с', 'т', 'у', 'ф', 'х'],
        ['ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э'],
        ['ю', 'я']
    ]

    template_drawer = TemplateDrawer(template_dir="templates", base_dir="symbols_dataset", sm_dir="sm_dig", cap_dir="cap", sym_dir="symbols")
    out_dir = "filled_templates"
    for i in tqdm(range(161)):
        var_dir = f"{i:04d}"
        os.makedirs(os.path.join(out_dir, var_dir), exist_ok=True)
        for j in range(9):
            first_page = template_drawer.fill_page("Calligraphr-Template_1.png", first_img_letters, i, j)
            second_page = template_drawer.fill_page("Calligraphr-Template_2.png", second_img_letters, i, j)
            cv2.imwrite(os.path.join(out_dir, var_dir, f"{j:02d}_1.png"), first_page)
            cv2.imwrite(os.path.join(out_dir, var_dir, f"{j:02d}_2.png"), second_page)

        images = [Image.open(os.path.join(out_dir, var_dir, f"{j:02d}_1.png")) for j in range(6)]
        images[0].save(os.path.join(out_dir, var_dir, f"{i:04d}_1.pdf"), "PDF", resolution=100.0, save_all=True, append_images=images[1:])
        images = [Image.open(os.path.join(out_dir, var_dir, f"{j:02d}_2.png")) for j in range(6)]
        images[0].save(os.path.join(out_dir, var_dir, f"{i:04d}_2.pdf"), "PDF", resolution=100.0, save_all=True, append_images=images[1:])
        images = [Image.open(os.path.join(out_dir, var_dir, f"{j:02d}_1.png")) for j in range(6, 9)] + \
                 [Image.open(os.path.join(out_dir, var_dir, f"{j:02d}_2.png")) for j in range(6, 9)]
        images[0].save(os.path.join(out_dir, var_dir, f"{i:04d}_3.pdf"), "PDF", resolution=100.0, save_all=True, append_images=images[1:])
