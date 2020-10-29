import os
import re
import string
import threading

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont
from matplotlib import font_manager

from Vision import Annotation, Sample
from Vision.io_managers import Manager
from Vision.sample import Sample_List, Sample_Generator
from Vision.shapes import Box
from Vision.utils import DefaultDictAddMissing
from Vision.utils.parallelisation import threadsafe_generator


def has_glyphs(font_path, alphabet):
    try:
        font = TTFont(os.path.join(font_path))
        for table in font['cmap'].tables:
            keys = table.cmap.keys()
            for glyph in alphabet:
                if not ord(glyph) in keys:
                    return False
        return True
    except Exception as e:
        return False


def transform_color(color, transform):
    return tuple(np.squeeze(cv2.cvtColor(np.array([[color]], dtype=np.uint8), transform)))


class SampleTextSynthesiser(Manager):
    """Sintetizador de imagenes con texto"""

    lock = threading.Lock()
    id_count = 0

    def __init__(self,
                 base_name: str,
                 n_batches: int,
                 batch_size: int,
                 alphabet=string.digits,
                 r_pattern=r".*",
                 r_noise_pattern=r"^$",
                 noise_tag="<n>",
                 use_system_fonts=True,
                 fonts=None,
                 text_length_range=(4, 7),
                 bg_text_ratio=1.5,
                 font_size_range=(12, 33),
                 black_n_white_chance=0.2,
                 color_perturbation=(60, 200),
                 noise_intensity_range=(3, 8),
                 noise_turbulence_range=(2, 21)
                 ):
        self.base_name = base_name
        self.noise_tag = noise_tag
        self.n_batches = n_batches
        self.r_noise_pattern = r_noise_pattern
        self.r_pattern = r_pattern
        self.text_lenght_range = text_length_range
        self.black_n_white_chance = black_n_white_chance
        self.noise_turbulence_range = noise_turbulence_range
        self.color_perturbation = color_perturbation
        self.noise_intensity_range = noise_intensity_range
        self.font_size_range = font_size_range
        self.bg_text_ratio = bg_text_ratio
        self.alphabet = list(alphabet)
        self.batch_size = batch_size
        if fonts is None:
            fonts = []
        self.fonts = fonts
        self.use_system_fonts = use_system_fonts
        if use_system_fonts:
            self.fonts += font_manager.findSystemFonts()
        self.valid_fonts = DefaultDictAddMissing(lambda path: has_glyphs(path, self.alphabet))
        assert len(fonts) > 0, "Debes especificar almenos el path a una fuente en formato TTF o OTF"

        super(SampleTextSynthesiser, self).__init__(None, None, [i for i in range(n_batches * batch_size)],
                                                    batch_size, False, 0)

    def clone(self) -> 'SampleTextSynthesiser':
        return SampleTextSynthesiser(
            base_name=self.base_name,
            n_batches=self.n_batches,
            batch_size=self.batch_size,
            alphabet=self.alphabet,
            r_pattern=self.r_pattern,
            r_noise_pattern=self.r_noise_pattern,
            noise_tag=self.noise_tag,
            use_system_fonts=self.use_system_fonts,
            fonts=self.fonts,
            text_length_range=self.text_lenght_range,
            bg_text_ratio=self.bg_text_ratio,
            font_size_range=self.font_size_range,
            black_n_white_chance=self.black_n_white_chance,
            color_perturbation=self.color_perturbation,
            noise_intensity_range=self.noise_intensity_range,
            noise_turbulence_range=self.noise_turbulence_range
        )

    def read_sample(self, n: int) -> Sample:
        return self.get_sample()

    def write_sample(self, sample: Sample, write_image=False) -> int:
        return 0

    @threadsafe_generator
    def sample_generator(self, batch_size: int = 0) -> Sample_Generator:
        if batch_size > 0:
            self.set_batch_size(batch_size)
        for i in range(self.n_batches):
            yield self.__getitem__(i)

    def write_samples(self, samples: Sample_List, write_image=False) -> int:
        return 0

    def get_sample(self):
        bg_text_ratio = 2.0
        bg_color, fg_color = self._generate_colors()
        font = self._generate_font()
        text = self._generate_text()
        text_size = font.getsize(text)
        bg_size = (round(text_size[0] * bg_text_ratio), round(text_size[1] * bg_text_ratio))
        bg_img = self._get_blurred_bg(bg_color, bg_size[0], bg_size[1])
        img = Image.fromarray(bg_img)
        draw = ImageDraw.Draw(img)
        text_xy = (
            round(text_size[0] * (bg_text_ratio - 1) * 0.5),
            round(text_size[1] * (bg_text_ratio - 1) * 0.5),
        )
        draw.text(
            text_xy,
            text,
            font=font,
            fill=fg_color
        )
        annotations = self._get_annotations(text, text_xy, font)
        sample_id = SampleTextSynthesiser.__get_new_id()
        base_name = self.base_name
        sample = Sample(
            sample_id,
            f"synth-{base_name}-{sample_id}.jpg",
            img,
            annotations).zoom_to_annotations(
            grow_w=self.bg_text_ratio,
            grow_h=self.bg_text_ratio,
        )

        return sample.substitute_annotations(sample.filter_annotations(self.r_pattern))

    def _generate_text(self):
        text_size = np.random.randint(*self.text_lenght_range)
        text = ''.join(np.random.choice(self.alphabet, text_size).tolist())
        return text

    def _generate_font(self):
        is_valid = False
        while not is_valid:
            font_path = np.random.choice(self.fonts, 1)[0]
            is_valid = self.valid_fonts[font_path]
        fs = np.random.randint(*self.font_size_range)
        return ImageFont.truetype(font_path, size=fs)

    def _generate_colors(self):
        black_n_white = np.random.random() <= self.black_n_white_chance
        if black_n_white:
            if np.random.randn() <= 0:
                bg_color = (0, 0, 0)
                fg_color = (255, 255, 255)
            else:
                bg_color = (255, 255, 255)
                fg_color = (0, 0, 0)
        else:
            bg_color = np.random.randint((0, 40, 40), (360, 255, 255), 3)
            perturbation = np.random.randint(self.color_perturbation[0], self.color_perturbation[1], 3)
            perturbation *= np.random.choice([-1, 1], 3)
            fg_color = bg_color + perturbation.astype(np.uint8)
            bg_color = transform_color(bg_color, cv2.COLOR_HSV2BGR)
            fg_color = transform_color(fg_color, cv2.COLOR_HSV2BGR)

        return bg_color, fg_color

    def _get_blurred_bg(self, bg_color, width, height):
        bg_img = np.array(Image.new("RGB", (width, height), bg_color))
        sigma = np.random.randint(*self.noise_intensity_range)
        bg_img = self.add_texture(bg_img, sigma=sigma)
        return bg_img

    def add_texture(self, bg_img, sigma):
        """
        Consequently applies noise patterns to the original image from big to small.

        sigma: defines bounds of noise fluctuations
        """
        result = bg_img.astype(float)
        cols, rows, ch = bg_img.shape
        ratio = int(min(cols, rows))
        # Defines how quickly big patterns will be replaced with the small ones. The lower
        # value - the more iterations will be performed during texture generation.
        turbulence = np.random.randint(2, 20)
        while not ratio == 1:
            result += self._gausian_noise(cols, rows, ratio, sigma=sigma)
            ratio = (ratio // turbulence) or 1
        cut = np.clip(result, 0, 255)
        return cut.astype(np.uint8)

    @staticmethod
    def _gausian_noise(width: int, height: int, ratio=1, sigma=3.0):
        """
        The function generates an image, filled with gaussian nose. If ratio parameter is specified,
        noise will be generated for a lesser image and then gen will be upscaled to the original size.
        In that case noise will generate larger square patterns. To avoid multiple lines, the upscale
        uses interpolation.

        :param ratio: the size of generated noise "pixels"
        :param sigma: defines bounds of noise fluctuations
        """
        mean = 0

        h = int(height // ratio)
        w = int(width // ratio)

        result = np.random.normal(mean, sigma, (w, h, 3))
        if ratio > 1:
            result = cv2.resize(result, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        return result.reshape((width, height, 3))

    def _get_annotations(self, text, text_xy, font):
        annotations = list()
        for i, char in enumerate(text):
            bottom_1 = font.getsize(text[i])[1]
            right, bottom_2 = font.getsize(text[:i + 1])
            bottom = bottom_1 if bottom_1 < bottom_2 else bottom_2
            width, height = font.getmask(char).size
            width, height = width + 1, height + 1
            right += text_xy[0]
            bottom += text_xy[1]
            top = bottom - height
            left = right - width
            shape = Box(left, top, width, height)
            if re.match(self.r_noise_pattern, char):
                cls = self.noise_tag
            else:
                cls = char
            annotations.append(
                Annotation(shape, cls=cls)
            )

        return annotations

    @classmethod
    def __get_new_id(cls) -> int:
        """
        Devuelve un id nuevo y auto-incrementa el actual en 1.
        :return: Nuevo id
        """
        with cls.lock:
            sample_id = cls.id_count
            cls.id_count += 1
        return sample_id
