import os
import json

from flask import Flask


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__)

    from . import xai
    app.register_blueprint(xai.bp)

    return app
