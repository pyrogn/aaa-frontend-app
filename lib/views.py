import hashlib

import aiohttp
import numpy as np
from aiohttp.web import Response, View
from aiohttp.web_exceptions import HTTPBadRequest
from aiohttp_jinja2 import render_template

from lib.image import PolygonDrawer, image_to_img_src, open_image


def get_image_hash(image_file) -> str:
    image_file.seek(0)

    hasher = hashlib.sha256()

    chunk_size = 8192
    while chunk := image_file.read(chunk_size):
        hasher.update(chunk)

    image_file.seek(0)

    return hasher.hexdigest()


cache = {}


class IndexView(View):
    template = "index.html"

    async def get(self) -> Response:
        return render_template(self.template, self.request, {})

    async def post(self) -> Response:
        try:
            image_file, _ = await self.get_image_file()
            image_hash = get_image_hash(image_file)
            words_draw = self.process_image(image_file, image_hash)
            image_b64 = image_to_img_src(
                PolygonDrawer(open_image(image_file)).get_highlighted_image()
            )
            ctx = {"image": image_b64, "words": words_draw}
        except Exception as err:
            ctx = {"error": err}
        return render_template(self.template, self.request, ctx)

    async def get_image_file(self):
        form = await self.request.post()
        image_field = form["image"]
        if not isinstance(image_field, aiohttp.web.FileField):
            raise HTTPBadRequest(reason="No image file has been uploaded.")
        if image_field.content_type not in ("image/jpeg", "image/png"):
            raise HTTPBadRequest(reason="Unsupported image format.")
        return image_field.file, image_field.content_type

    def process_image(self, image_file, image_hash):
        if image_hash in cache:
            return cache[image_hash]
        image = open_image(image_file)
        model = self.request.app["model"]
        words_detected = model.readtext(np.array(image))
        cache[image_hash] = words_detected
        return self.draw_words(image, words_detected)

    def draw_words(self, image, words_detected):
        draw = PolygonDrawer(image)
        words_draw = []
        for coords, word, accuracy in words_detected:
            draw.highlight_word(coords, word)
            cropped_img = draw.crop(coords)
            cropped_img_b64 = image_to_img_src(cropped_img)
            words_draw.append(
                {
                    "image": cropped_img_b64,
                    "word": word,
                    "accuracy": accuracy,
                }
            )
        return words_draw
