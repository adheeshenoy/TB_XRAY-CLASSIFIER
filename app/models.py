from tensorflow.keras.models import load_model
from tensorflow.keras.losses import Loss
from tensorflow.nn import weighted_cross_entropy_with_logits
import tensorflow.keras.preprocessing.image as tkpi
from helper import resource_path

class weighted_loss(Loss):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = 2.0
    def call(self, y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight = self.weight)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'weight': self.weight
        })
        return config

# Load models
models = {
    'model_1': {'clf': load_model(resource_path('unweighted_model.h5')), 'threshold': 0.44772145},
    'model_2': {'clf': load_model(resource_path('weighted_model.h5'), custom_objects = {'weighted_loss' : weighted_loss}), 'threshold': 0.44156814},
}

def load_image(file_path, image_size = 128):
    '''Load images for model'''
    my_image = tkpi.load_img(file_path, target_size=(image_size, image_size))
    my_image = tkpi.img_to_array(my_image)
    my_image /= 255
    my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
    return my_image

def get_prediction(file_path, image_size = 128):
    img = load_image(file_path)
    return any(1 if i['clf'].predict(img)[0][0] >= i['threshold'] else 0 for i in models.values())
