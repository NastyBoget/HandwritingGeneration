allowed_symbols = 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯабвгдежзийклмнопрстуфхцчшщьыъэюя0123456789.!"%(),\-?:; '

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
