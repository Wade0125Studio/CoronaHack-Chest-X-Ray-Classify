import flask
import os
import datetime
import cv2
import numpy as np
from utils.config import Config
from utils.file import File, human_readable_size
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
  except RuntimeError as e:
    print(e)
model = tf.keras.models.load_model("./EfficientNetV2S.h5")
blueprint = flask.Blueprint('file', __name__)
allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."

@blueprint.route('/view/<path:path>')
def get_file(path):
    if "authorization" not in flask.session:
        return flask.redirect(flask.url_for('general.login'))
    if flask.session["authorization"] != Config.get_auth_token():
        return flask.jsonify({'error': 'unauthorized'}), 403

    file_path = os.path.join(Config.get_upload_dir(), path)
    if not os.path.isfile(file_path):
        return flask.jsonify({'error': 'file does not exist'}), 404
    
    file_delete_path = file_path.replace("files/", "/delete/")
    file_rename_path = file_path.replace("files/", "/rename/")
    file_extension = os.path.splitext(file_path)[1]
    file_date = os.path.getmtime(file_path)
    file_date = datetime.datetime.fromtimestamp(file_date).strftime('%Y-%m-%d %H:%M:%S')
    file_size = os.path.getsize(file_path)
    file_size = human_readable_size(file_size)
    print(file_extension)
    img = cv2.imread('./files/'+path)
    img = cv2.resize(img, (150,150), interpolation=cv2.INTER_AREA)
    img=img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
    preds=model.predict(img,batch_size=1)
    preds = np.argmax(preds,axis=1)
    labels={0: 'Normal', 1: 'Pnemonia'}
    predict =  [labels[k] for k in preds][0]
    print(predict)
    file = File(file_extension, file_path.replace("files/", "/view/raw/"), file_delete_path, file_rename_path, file_path.replace("files/", ""), file_date, file_size,predict)

    return flask.render_template('view.html', file=file, auth_token=flask.session["authorization"])

@blueprint.route('/view/raw/<path:path>')
def get_raw_file(path):
    if "authorization" not in flask.session:
        return flask.redirect(flask.url_for('general.login'))
    if flask.session["authorization"] != Config.get_auth_token():
        return flask.jsonify({'error': 'unauthorized'}), 403

    file_path = os.path.join(Config.get_upload_dir(), path)
    if not os.path.isfile(file_path):
        return flask.jsonify({'error': 'file does not exist'}), 404

    print(file_path)

    return flask.send_file(file_path)

@blueprint.route('/rename/<path:path>', methods=['POST'])
def rename_file(path):
    if "authorization" not in flask.session:
        return flask.redirect(flask.url_for('general.login'))
    if flask.session["authorization"] != Config.get_auth_token():
        return flask.jsonify({'error': 'unauthorized'}), 403

    file_path = os.path.join(Config.get_upload_dir(), path)
    if not os.path.isfile(file_path):
        return flask.jsonify({'error': 'file does not exist'}), 404

    new_name = flask.request.json["new_name"]
    if not new_name:
        return flask.jsonify({'error': 'no new name'}), 400

    if not all(c in allowed_chars for c in new_name):
        return flask.jsonify({'error': 'invalid characters in name'}), 400

    os.rename(file_path, os.path.join(Config.get_upload_dir(), new_name))
    return flask.jsonify({'success': True}), 200

@blueprint.route('/delete/<path:path>')
def delete_file(path):
    if "authorization" not in flask.session:
        return flask.redirect(flask.url_for('general.login'))
    if flask.session["authorization"] != Config.get_auth_token():
        return flask.jsonify({'error': 'unauthorized'}), 403

    try:
        os.remove(os.path.join(Config.get_upload_dir(), path))
        return flask.jsonify({'success': True}), 200

    except FileNotFoundError: return flask.jsonify({'error': 'file not found'}), 404
    except PermissionError: return flask.jsonify({'error': 'permission denied'}), 403
    except Exception as e: return flask.jsonify({'error': str(e)}), 500