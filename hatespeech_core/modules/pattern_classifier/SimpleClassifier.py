import os

import numpy as np
import pandas as pd

from .PatternVectorizer import PatternVectorizer

class SimpleClassifier:
    
  def __init__(self, scoreMatrix, classes=None):
    # scoreMatrix each represent the vector of one pattern
    self.scoreMatrix = scoreMatrix
    
    # Default set classes as integer
    if np.all(classes is None):
      classes = list(range(scoreMatrix.shape[1]))
    if len(classes) < scoreMatrix.shape[1]:
      raise ValueError('Classes dimentions does not fit with the score matrix')
    
    self.classes = classes  
          
  def get_top_classes(self, documentPatternVectors, n=1, ascending=True):
    d = 1 if ascending else -1
    return [[self.classes[c] for c in vect[::d][:n]] for vect in self.get_emotion_score(documentPatternVectors).argsort()]
    
  def get_max_score_class(self, documentPatternVectors):
    return [self.classes[c] for c in self.get_emotion_score(documentPatternVectors).argmax(axis=1)]
    
  def get_min_score_class(self, documentPatternVectors):
    return [self.classes[c] for c in self.get_emotion_score(documentPatternVectors).argmin(axis=1)]
  
  def get_emotion_score(self, documentPatternVectors):
    return documentPatternVectors.dot(self.scoreMatrix)  
  
  
  @classmethod
  def load_from_folder(cls, patternFolderPath, rank='average', ascending=False):
    patterns_df = pd.DataFrame()
    emotions = []
    for filename in os.listdir(patternFolderPath):
      emotion = filename.replace('tficf_','')
      col = ['pattern', emotion]
      emotions.append(emotion) 
      tmp = pd.read_table(os.path.join(patternFolderPath, filename), header=None, names=col)
      if rank:
        tmp[emotion] = tmp[emotion].rank(method=rank, ascending=ascending, pct=True)
      if len(patterns_df) == 0:
        patterns_df = tmp
      else:
        patterns_df = pd.merge(patterns_df, tmp ,on='pattern')
    
    pv = PatternVectorizer(list(patterns_df.pattern))
    new_cls = cls(patterns_df[emotions].values, classes=emotions)
    return (pv, new_cls)
