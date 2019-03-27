import numpy as np

# A simplified version of TensorFlow's DataSet class
class DataSet(object):

  def __init__(self,
               points,
               labels):

    if points is not None:
        self._num_examples = points.shape[0]

        self._points = points
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
  @property
  def points(self):
    return self._points

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._points = self._points[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._points[start:end], self._labels[start:end]