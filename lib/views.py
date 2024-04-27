import aiohttp
from aiohttp.web import Response
from aiohttp.web import View

from aiohttp_jinja2 import render_template
from aiohttp.web_exceptions import HTTPBadRequest

from lib.image import image_to_img_src
from lib.image import PolygonDrawer
from lib.image import open_image
import hashlib
import numpy as np


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
        ctx = {}
        return render_template(self.template, self.request, ctx)

    async def post(self) -> Response:
        try:
            form = await self.request.post()
            image_field = form["image"]
            if not isinstance(image_field, aiohttp.web.FileField):
                raise HTTPBadRequest(reason="No image file has been uploaded.")

            image_file = image_field.file
            content_type = image_field.content_type
            if content_type not in ("image/jpeg", "image/png"):
                raise HTTPBadRequest(reason="Unsupported image format.")

            image_hash = get_image_hash(image_file)
            if image_hash in cache:
                words_detected = cache[image_hash]
            else:
                image = open_image(image_file)
                model = self.request.app["model"]
                words_detected = model.readtext(np.array(image))
                cache[image_hash] = words_detected

            words_draw = []
            image = open_image(image_file)
            draw = PolygonDrawer(image)
            model = self.request.app["model"]
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

            image_b64 = image_to_img_src(draw.get_highlighted_image())
            ctx = {"image": image_b64, "words": words_draw}
        except Exception as err:
            ctx = {"error": err}
        return render_template(self.template, self.request, ctx)
