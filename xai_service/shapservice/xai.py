import io
import os
import json
from flask import (
    Blueprint, request, jsonify, send_file
)
import numpy as np
import shap

from . import shap_helper

bp = Blueprint('shapservice', __name__, url_prefix='/shapservice')


@bp.route('/', methods=['POST','GET'])
def pred():
    if request.method == 'POST':

        return 0

    elif request.method == 'GET':
        return 0

