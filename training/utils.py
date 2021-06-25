import os
from datetime import datetime
import uuid
import tensorflow as tf


def save_model(model, config, save_directory=None):
    models_dir = save_directory or 'saved_models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model_id = f'{models_dir}/{datetime.now()}---{uuid.uuid4()}'
    os.makedirs(model_id)

    with open(f'{model_id}/info.json', 'w') as f:
        import json
        json.dump(config, f)

    tf.keras.models.save_model(model, f'{model_id}/model')
