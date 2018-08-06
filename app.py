import torch as th
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.gen
import tornado.web
import pickle
import os
import sys
import logging
import json
from mnist_model import ToyNet

from tornado.options import define, options

define('port', default=6666, type=int)

class InferenceHandler(tornado.web.RequestHandler):
    def initialize(self, model_path, model_obj=None, cuda=False):
        self.cuda = cuda
        if model_obj is None:
            self.model = th.load(model_path)
        else:
            self.model = model_obj
            model_dict = self.model.state_dict()
            static_weight = th.load(model_path)
            static_weight = {k: v for k, v in static_weight.items() if k in model_dict}
            model_dict.update(static_weight)

            self.model.load_state_dict(model_dict)
        if self.cuda:
            self.model.cuda()
        self.model.eval()

    def _inference(self, input_data):
        res = self.model(input_data)
        return res

    @tornado.web.asynchronous
    @tornado.gen.engine
    def post(self):
        data = pickle.loads(self.request.body)
        if self.cuda:
            data.cuda()
            res = self._inference(data).cpu()
        else:
            res = self._inference(data)
        pickled = pickle.dumps(res)
        self.write(pickled)
        self.finish()


if __name__ == '__main__':
    tornado.options.parse_command_line()
    model_path = './model/toynet.pth.tar'
    model_obj = ToyNet()
    app = tornado.web.Application(
        [(r'/', InferenceHandler,
         dict(model_path=model_path,
              model_obj=model_obj,
              cuda=False))], debug=True
    )

    app.listen(options.port, address='0.0.0.0')
    tornado.ioloop.IOLoop.instance().start()

