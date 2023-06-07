import flask
import os
import datetime
import hashlib
import cv2
import numpy as np
from utils.file import File, human_readable_size
from utils.config import Config
import tensorflow as tf
model = tf.keras.models.load_model("./EfficientNetV2S.h5")
blueprint = flask.Blueprint('general', __name__)

@blueprint.route('/')
def index():
    if "authorization" not in flask.session:
        logged_in = False
    else:
        logged_in = flask.session["authorization"] == Config.get_auth_token()

    return flask.render_template('index.html', logged_in=logged_in)

@blueprint.route('/login', methods=['GET', 'POST'])
def login():
    if "authorization" not in flask.request.form or flask.request.form['authorization'] != Config.get_auth_token() or flask.request.method == 'GET':
        return flask.redirect(flask.url_for('general.index'))

    flask.session['authorization'] = Config.get_auth_token()
    return flask.redirect(flask.url_for('general.files'))

@blueprint.route('/logout')
def logout():
    flask.session.pop('authorization', None)
    return flask.redirect(flask.url_for('general.index'))

@blueprint.route('/files', methods=['GET'])
def files():
    if flask.session.get('authorization') != Config.get_auth_token():
        return flask.redirect(flask.url_for('general.login'))

    files = []
    for file in os.listdir(Config.get_upload_dir()):
        file_path = "/view/" + os.path.join(Config.get_upload_dir(), file).replace(Config.get_upload_dir(), "")
        file_delete_path = file_path.replace("/view/", "/delete/")
        file_rename_path = file_path.replace("/view/", "/rename/")
        file_extension = os.path.splitext(file)[1]
        file_date = os.path.getmtime(os.path.join(Config.get_upload_dir(), file))
        file_date = datetime.datetime.fromtimestamp(file_date).strftime('%Y-%m-%d %H:%M:%S')
        file_size = os.path.getsize(os.path.join(Config.get_upload_dir(), file))
        file_size = human_readable_size(file_size)
        print(file)
        img = cv2.imread('./files/'+file)
        img = cv2.resize(img, (150,150), interpolation=cv2.INTER_AREA)
        img=img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
        preds=model.predict(img,batch_size=1)
        preds = np.argmax(preds,axis=1)
        labels={0: 'Normal', 1: 'Pnemonia'}
        predict =  [labels[k] for k in preds][0]
        files.append(File(file_extension, file_path, file_delete_path, file_rename_path, file, file_date, file_size,predict))

    return flask.render_template('files.html', files=files, auth_token_hash=hashlib.md5(Config.get_auth_token().encode('utf-8')).hexdigest(), default_auth_token_hash="b0439fae31f8cbba6294af86234d5a28")