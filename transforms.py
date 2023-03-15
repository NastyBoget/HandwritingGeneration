import math
import cv2
import random
import numpy as np
import skimage
from albumentations import augmentations
from augmixations import HandWrittenBlot
from numpy.random import default_rng


class UseWithProb:
    def __init__(self, transform, prob=0.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = self.transform(image)
        return image


class OneOf:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        return random.choice(self.transforms)(image)


class Rotate:
    def __init__(self, max_ang, prob):
        self.aug = augmentations.geometric.rotate.Rotate(limit=max_ang, p=prob)

    def __call__(self, img):
        augmented = self.aug(image=img)
        return augmented['image']


class SafeRotate:
    def __init__(self, max_ang, prob):
        self.aug = augmentations.geometric.rotate.SafeRotate(
            limit=max_ang, p=prob)

    def __call__(self, img):
        augmented = self.aug(image=img)
        return augmented['image']


class RandomCrop:
    def __init__(self, rnd_crop_min, rnd_crop_max=1):
        self.factor_max = rnd_crop_max
        self.factor_min = rnd_crop_min

    def __call__(self, img):
        factor = random.uniform(self.factor_min, self.factor_max)
        size = (
            int(img.shape[1]*factor),
            int(img.shape[0]*factor)
        )
        img, x1, y1 = self.__random_crop(img, size)
        return img

    def __img_crop(self, img, bbox):
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def __random_crop(self, img, size):
        tw = size[0]
        th = size[1]
        h, w = img.shape[:2]
        if ((w - tw) > 0) and ((h - th) > 0):
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
        else:
            x1 = 0
            y1 = 0
        img_return = self.__img_crop(img, (x1, y1, x1 + tw, y1 + th))
        return img_return, x1, y1


class RotateAndCrop:
    """Random image rotate around the image center

    Args:
        max_ang (float): Max angle of rotation in deg
    """

    def __init__(self, max_ang=0):
        self.max_ang = max_ang

    def __call__(self, img):
        h, w, _ = img.shape

        ang = np.random.uniform(-self.max_ang, self.max_ang)
        M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1)
        img = cv2.warpAffine(img, M, (w, h))

        w_cropped, h_cropped = self.__largest_rotated_rect(w, h, math.radians(ang))
        # to fix cases of too small or negative image height when cropping
        h_cropped = max(h_cropped, 10)
        img = self.__crop_around_center(img, w_cropped, h_cropped)
        return img

    def __largest_rotated_rect(self, w, h, angle):
        """
        https://stackoverflow.com/a/16770343
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle within the rotated rectangle.
        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
        Converted to Python by Aaron Snoswell
        """

        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )

    def __crop_around_center(self, image, width, height):
        """
        https://stackoverflow.com/a/16770343
        Given a NumPy / OpenCV 2 image, crops it to the given width and height,
        around its centre point
        """

        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if (width > image_size[0]):
            width = image_size[0]

        if (height > image_size[1]):
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]


class CLAHE:
    def __init__(self, prob):
        self.aug = augmentations.transforms.CLAHE(p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class GaussNoise:
    def __init__(self, prob):
        self.aug = augmentations.transforms.GaussNoise(
            var_limit=100, p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class ISONoise:
    def __init__(self, prob):
        self.aug = augmentations.transforms.ISONoise(
            p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class MultiplicativeNoise:
    def __init__(self, prob):
        self.aug = augmentations.transforms.MultiplicativeNoise(
            multiplier=(0.85, 1.15), p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class ImageCompression:
    def __init__(self, prob):
        self.aug = augmentations.transforms.ImageCompression(
            quality_lower=60, quality_upper=90, p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class Sharpen:
    def __init__(self, prob):
        self.aug = augmentations.Sharpen(
            p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class ElasticTransform:
    def __init__(self, prob):
        self.aug = augmentations.geometric.transforms.ElasticTransform(
            alpha_affine=2.5, p=prob)

    def __call__(self, img):
        augmented = self.aug(image=img)
        return augmented['image']


class GridDistortion:
    def __init__(self, prob):
        self.aug = augmentations.transforms.GridDistortion(p=prob)

    def __call__(self, img):
        augmented = self.aug(image=img)
        return augmented['image']


class OpticalDistortion:
    def __init__(self, prob):
        self.aug = augmentations.transforms.OpticalDistortion(
            distort_limit=0.2, p=prob)

    def __call__(self, img):
        augmented = self.aug(image=img)
        return augmented['image']


class Perspective:
    def __init__(self, prob):
        self.aug = augmentations.geometric.transforms.Perspective(
            pad_mode=2, fit_output=True, p=prob)

    def __call__(self, img):
        augmented = self.aug(image=img)
        return augmented['image']


class ChannelDropout:
    def __init__(self, prob):
        self.aug = augmentations.ChannelDropout(p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class ChannelShuffle:
    def __init__(self, prob):
        self.aug = augmentations.transforms.ChannelShuffle(p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class RGBShift:
    def __init__(self, prob):
        self.aug = augmentations.transforms.RGBShift(p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class ToGray:
    def __init__(self, prob):
        self.aug = augmentations.transforms.ToGray(p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class ToSepia:
    def __init__(self, prob):
        self.aug = augmentations.transforms.ToSepia(p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class RandomBrightnessContrast:
    def __init__(self, prob):
        self.aug = augmentations.transforms.RandomBrightnessContrast(p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class RandomSnow:
    def __init__(self, prob):
        self.aug = augmentations.transforms.RandomSnow(
            brightness_coeff=1.5, p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class HueSaturationValue:
    def __init__(self, prob):
        self.aug = augmentations.transforms.HueSaturationValue(p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class RandomShadow:

    def __call__(self, image, mask=None):
        row, col, ch = image.shape
        # We take a random point at the top for the x coordinate and then
        # another random x-coordinate at the bottom and join them to create
        # a shadow zone on the image.
        top_y = col * np.random.uniform()
        top_x = 0
        bot_x = row
        bot_y = col * np.random.uniform()
        img_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        shadow_mask = 0 * img_hls[:, :, 1]
        X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
        Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

        shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1

        random_bright = .25 + .7 * np.random.uniform()
        cond0 = shadow_mask == 0
        cond1 = shadow_mask == 1

        if np.random.randint(2) == 1:
            img_hls[:, :, 1][cond1] = img_hls[:, :, 1][cond1] * random_bright
        else:
            img_hls[:, :, 1][cond0] = img_hls[:, :, 1][cond0] * random_bright
        image = cv2.cvtColor(img_hls, cv2.COLOR_HLS2RGB)

        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)

        if mask is not None:
            return image, mask
        else:
            return image


class RandomGamma:
    def __init__(self, prob):
        self.aug = augmentations.transforms.RandomGamma(
            gamma_limit=(50, 150), p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class MotionBlur:
    def __init__(self, prob):
        self.aug = augmentations.transforms.MotionBlur(
            blur_limit=7, p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class MedianBlur:
    def __init__(self, prob):
        self.aug = augmentations.transforms.MedianBlur(
            blur_limit=5, p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class GlassBlur:
    def __init__(self, prob):
        self.aug = augmentations.transforms.GlassBlur(
            sigma=0.7, max_delta=2, p=prob)

    def __call__(self, img):
        img = self.aug(image=img)['image']
        return img


class ChangeWidth:
    def __call__(self, img):
        proportion = random.uniform(0.5, 2.)
        new_width = int(img.shape[1] * proportion)
        img = cv2.resize(img, (new_width, img.shape[0]), interpolation=cv2.INTER_AREA)
        return img


class Erosion:

    def __call__(self, img):
        img = cv2.erode(img, np.ones(2, np.uint8), iterations=1)
        return img


class Dilation:

    def __call__(self, img):
        img = cv2.dilate(img, np.ones(2, np.uint8), iterations=1)
        return img


class RandomBlot:
    def __call__(self, img):
        blots_num = random.randint(1, 3)
        b = HandWrittenBlot(
            {'x': (None, None), 'y': (None, None), 'h': (int(img.shape[0] * 0.1), int(img.shape[0] * 0.5)),
             'w': (int(img.shape[1] * 0.1), int(img.shape[1] * 0.2))},  # noqa
            {'incline': (10, 50), 'intensivity': (0.75, 0.75), 'transparency': (0.05, 0.4), 'count': blots_num}
        )
        return b.apply(img)


class GradientBackground:

    def __call__(self, img):
        light = random.choice([True, False])
        rotate = random.randint(0, 3)
        color = random.randint(100, 200)
        background_img = np.ones((img.shape[1], img.shape[1], 3), dtype=np.uint8) * 255
        gradient_mask = np.rot90(np.repeat(np.tile(np.linspace(1, 0, background_img.shape[0]),
                                                   (background_img.shape[0], 1))[:, :, np.newaxis], 3, axis=2), rotate)
        background_img[:, :, :] = gradient_mask * background_img + (1 - gradient_mask) * color
        background_img = 255 - cv2.resize(background_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        background_img = background_img.astype(np.uint8)

        if light:
            return cv2.add(img, background_img)
        else:
            img = cv2.add(255 - img, background_img)
            return 255 - img


class RandomStains:

    def __call__(self, img):
        light = random.choice([True, False])
        color = random.randint(100, 200)
        rng = default_rng(seed=random.randint(0, 1000))
        noise = rng.integers(0, 255, (img.shape[0], img.shape[1]), np.uint8, True)
        blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)
        stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255))
        thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.merge([mask, mask, mask])
        result = np.where(mask > 0, 255 - color, 0).astype(np.uint8)
        result = result.astype(np.uint8)

        if light:
            return cv2.add(img, result)
        else:
            img = cv2.add(255 - img, result)
            return 255 - img


class RandomBlurredStains:

    def __call__(self, img):
        light = random.choice([True, False])
        max_color = random.randint(100, 200)
        rng = default_rng(seed=random.randint(0, 1000))
        noise = rng.integers(0, 255, (img.shape[0], img.shape[1]), np.uint8, True)
        blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)

        background_img = cv2.merge([blur, blur, blur]) / 255.
        background_img = ((background_img - background_img.min()) * max_color / (background_img.max() - background_img.min())).astype(np.uint8)
        background_img = background_img.astype(np.uint8)

        if light:
            return cv2.add(img, background_img)
        else:
            img = cv2.add(255 - img, background_img)
            return 255 - img


class CutCharacters:

    def __call__(self, img):
        under = random.choice([True, False])
        img = 255 - img
        y = int(0.3 * img.shape[0])
        cut_img = img[:y, :] if under else img[-y:, :]
        cut_img = np.roll(cut_img, random.randint(0, cut_img.shape[1]), axis=1)

        if under:
            img[-y:, :] = cv2.add(img[-y:, :], cut_img)
        else:
            img[:y, :] = cv2.add(img[:y, :], cut_img)
        return 255 - img


def get_cyrillic_transforms(prob):
    transforms = [
        UseWithProb(CutCharacters(), prob),
        UseWithProb(OneOf([RandomBlot(), RandomStains(), RandomBlurredStains(), GradientBackground(), RandomShadow()]), prob),
        UseWithProb(OneOf([Dilation(), Erosion()]), prob),
        OneOf([CLAHE(prob), GaussNoise(prob), ISONoise(prob), MultiplicativeNoise(prob), ImageCompression(prob), Sharpen(prob), MotionBlur(prob), MedianBlur(prob)]),
        OneOf([
            RandomCrop(rnd_crop_min=0.80),
            RotateAndCrop(2),
            Rotate(2, prob),
            SafeRotate(5, prob),
            ElasticTransform(prob),
            GridDistortion(prob),
            OpticalDistortion(prob),
            Perspective(prob),
            ChangeWidth()
        ]),
        OneOf([RandomBrightnessContrast(prob), RandomGamma(prob), HueSaturationValue(prob), RandomSnow(prob)]),
    ]
    return transforms


def get_hkr_transforms(prob):
    transforms = [
        UseWithProb(OneOf([Dilation(), Erosion()]), prob),
        OneOf([CLAHE(prob), GaussNoise(prob), ISONoise(prob), MultiplicativeNoise(prob), ImageCompression(prob), Sharpen(prob), MotionBlur(prob), MedianBlur(prob)]),
        OneOf([
            RandomCrop(rnd_crop_min=0.80),
            RotateAndCrop(2),
            Rotate(2, prob),
            SafeRotate(5, prob),
            ElasticTransform(prob),
            GridDistortion(prob),
            OpticalDistortion(prob),
            Perspective(prob),
            ChangeWidth()
        ]),
        OneOf([RandomBrightnessContrast(prob), RandomGamma(prob), HueSaturationValue(prob), RandomSnow(prob)]),
    ]
    return transforms
