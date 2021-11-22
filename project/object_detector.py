import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow as tf1
tf1.compat.v1.logging.set_verbosity(tf1.compat.v1.logging.ERROR)

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class TFObjectDetector:
  """
  detects objects using a given TensorFlow model
  """

  def __init__(self, model_path, labelmap_path):
    """
    creates a new TFObjectDetector

    arguments:
      - model_path: path to directory containing saved TensorFlow model
      - labelmap_path: path to the map of indices to category names
    """
    self.model_path = model_path
    self.labelmap_path = labelmap_path
    self.loadModel()
    self.loadLabelmap()

  def detectObjectsInImage(self, image):
    """
    returns object-detection results for the given image

    arguments:
      - image: 2D array of RGB triplets (each value in range [0, 255])
    """
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]
    output_dict = self.model(input_tensor)
    
    return output_dict

  def loadLabelmap(self):
    """
    loads the map of indices to category names
    """
    self.category_index = label_map_util.create_category_index_from_labelmap(
      self.labelmap_path, use_display_name=True)
    

  def loadModel(self):
    """
    loads the TensorFlow object-detection model
    """
    model_dir = pathlib.Path(self.model_path)
    model = tf.saved_model.load(str(model_dir))
    self.model = model.signatures['serving_default']
