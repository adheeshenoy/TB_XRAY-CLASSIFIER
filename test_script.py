import tensorflow.keras as keras
import pandas as pd 
import numpy as np 
import tensorflow as tf 
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, classification_report
import matplotlib.pyplot as plt

class weighted_loss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = 2.5
    def call(self, y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight = self.weight)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'weight': self.weight
        })
        return config
        

def threshold_pred(predictions, threshold = 0.5):
    return [0 if i[0] < threshold else 1 for i in predictions]

def plot_roc_curve(x, y, title):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
        title= title)
    ax.grid()
    fig.savefig(f"images/{title}.png")

def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    
    fig.savefig(f"images/{title}.png")

def print_info(data, gt, preds):
    '''print overall accuracy and confusion matrix'''
    print(data.title())
    print(accuracy_score(gt, preds)
    print(confusion_matrix(gt, preds)
    print('=' * 10)

def main(image_size, model_name):
    '''test model against validation and test dataset'''
    val_batch = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,).flow_from_directory('val', target_size = (image_size, image_size), batch_size = 32, class_mode='binary', shuffle = False)
    test_batch = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,).flow_from_directory('test', target_size = (image_size, image_size), batch_size = 32, class_mode='binary', shuffle = False)

    model = keras.models.load_model(f'{model_name}.h5', custom_objects = {'weighted_loss' : weighted_loss})
    predictions = model.predict(x = val_batch, verbose = 0)

    fpr, tpr, thresholds = roc_curve(val_batch.classes, predictions[:, 0])
    plot_roc_curve(fpr, tpr, f'{model_name.title()}_ROC')
    
    # Compute threshold
    th = thresholds[np.argmin(np.abs(fpr+tpr-1))]
    print('Threshold:', th)
    
    print_info('validation', val_batch.classes, threshold_pred(predictions, th))
    
    predictions = model.predict(x = test_batch, verbose = 0)
    print_info('test', test_batch.classes, threshold_pred(predictions, th))
    
def vote(image_size):
    '''
    Ensemble voting method to get prediction from 2 models
    outputs results from test dataset
    '''
    
    test_batch = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,).flow_from_directory('val', target_size = (image_size, image_size), batch_size = 32, class_mode='binary', shuffle = False)

    model = keras.models.load_model('weighted_model.h5', custom_objects = {'weighted_loss' : weighted_loss})
    weighted_predictions = threshold_pred(model.predict(x = test_batch, verbose = 0), 0.44156814)
    
    model = keras.models.load_model('unweighted_model.h5', custom_objects = {'weighted_loss' : weighted_loss})
    unweighted_predictions = threshold_pred(model.predict(x = test_batch, verbose = 0), 0.44772145)
    
    preds = [1 if i or j else 0 for i,j in zip(weighted_predictions, unweighted_predictions)]
    
    print_info('test', test_batch.classes, preds)
        
    cm = confusion_matrix(test_batch.classes, preds)
    plot_confusion_matrix(cm, 'final_confusion_matrix')

    print(classification_report(test_batch.classes, preds))


if __name__ == '__main__':
    # main(128, 'weighted_model')
    vote(128)
