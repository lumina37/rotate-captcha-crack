import functools
import io
import json

from aiohttp import web
from PIL import Image

from rotate_captcha_crack.logging import RCCLogger

logger = RCCLogger()
routes = web.RouteTableDef()

dumps = functools.partial(json.dumps, separators=(',', ':'))


@routes.get('/')
async def hello(request: web.Request):
    resp = {'err': {'code': 0, 'msg': 'success'}}

    try:
        multipart = await request.multipart()
        image_part = await multipart.next()
        image_bytes = await image_part.read()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as err:
        resp['err']['code'] = 0x01
        resp['err']['msg'] = str(err)
        return web.json_response(resp, status=400, dumps=dumps)

    resp['pred'] = float(image.height)

    return web.json_response(resp, dumps=dumps)


app = web.Application()
app.add_routes(routes)
web.run_app(app, port=4396, access_log_format='%a "%r" %s %b', access_log=logger)
