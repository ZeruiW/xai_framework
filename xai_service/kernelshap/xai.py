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









def pred(X):
    paper_data = []
    for x in X:

        paper_data.append(decode_paper(categorical_name, x))

    return get_scores("Hardware Architecture", paper_data, ptf=False)






@bp.route('/', methods=['POST','GET'])
def pred():
    if request.method == 'POST':





        predict_function = lambda z: pred(z)
        explainer = shap.KernelExplainer(predict_function, df)
        shap_values = explainer.shap_values(df)
        base_value = explainer.expected_value
        return jsonify(shap_values, base_value)

    elif request.method == 'GET':
        return 0

