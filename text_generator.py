import re

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
a

class TextGenerator:
    not_allowed_symbols = re.compile(
        r'[^АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯабвгдежзийклмнопрстуфхцчшщьыъэюя0123456789.!"%(),\-?:; ]')

    title_url = "https://ru.wikipedia.org/w/api.php?origin=*&action=query&format=json&list=random&rnlimit=1&rnnamespace=0"
    article_url = "https://ru.wikipedia.org/w/api.php?origin=*&action=parse&format=json&page={title}&prop=text"

    def get_random_text(self) -> str:
        article_text_fixed = ""

        while len(article_text_fixed) == 0:
            try:
                # 1 - Get random title of the article in Wikipedia
                title_result = requests.post(self.title_url)
                title_result_dict = title_result.json()
                title = title_result_dict["query"]["random"][0]["title"]

                # 2 - Get text the article
                article_result = requests.post(self.article_url.format(title=title))
                article_result_dict = article_result.json()
                article = article_result_dict["parse"]["text"]['*']
                bs = BeautifulSoup(article, 'html.parser')
                article_text = bs.get_text()

                # 3 - Clear text of the article from unused symbols
                article_text_fixed = re.sub(r"[«»]", '"', article_text)
                article_text_fixed = re.sub(self.not_allowed_symbols, " ", article_text_fixed)
                article_text_fixed = re.sub(r"\s[.!\"%(),\-?:;]\s", " ", article_text_fixed)
                article_text_fixed = re.sub(r"\s+", " ", article_text_fixed)
            except:
                article_text_fixed = ""

        return article_text_fixed


if __name__ == "__main__":
    gen = TextGenerator()
    texts = []
    for i in tqdm(range(20)):
        texts.append(gen.get_random_text())

    with open("images/texts.txt", "w") as f:
        f.write("\n".join(texts))
