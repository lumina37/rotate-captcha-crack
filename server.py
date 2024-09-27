import argparse
import functools
import io
import json

import torch
from aiohttp import web
from PIL import Image

from rotate_captcha_crack.common import device
from rotate_captcha_crack.const import DEFAULT_CLS_NUM
from rotate_captcha_crack.logging import RCCLogger
from rotate_captcha_crack.model import RotNetR, WhereIsMyModel
from rotate_captcha_crack.utils import process_captcha

logger = RCCLogger()
routes = web.RouteTableDef()

dumps = functools.partial(json.dumps, separators=(',', ':'))

parser = argparse.ArgumentParser()
parser.add_argument("--index", "-i", type=int, default=-1, help="Use which index")
opts = parser.parse_args()

cls_num = DEFAULT_CLS_NUM
model = RotNetR(cls_num=cls_num, train=False)
model_path = WhereIsMyModel(model).with_index(opts.index).model_dir / "best.pth"
model.load_state_dict(torch.load(model_path, weights_only=True))
model = model.to(device=device)
model.eval()


@routes.post('/')
async def handler(request: web.Request):
    resp = {'err': {'code': 0}}

    try:
        img_bytes = await request.read()
        img = Image.open(io.BytesIO(img_bytes))

        with torch.no_grad():
            img_ts = process_captcha(img)
            img_ts = img_ts.to(device=device)
            predict = model.predict(img_ts) * 360
            resp['pred'] = predict

    except Exception as err:
        resp['err']['code'] = 0x0001
        resp['err']['msg'] = str(err)
        return web.json_response(resp, status=400, dumps=dumps)

    return web.json_response(resp, dumps=dumps)


app = web.Application()
app.add_routes(routes)
web.run_app(app, port=4396, access_log_format='%a "%r" %s %b', access_log=logger)
