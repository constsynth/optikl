"""
Hello! This is constantly evolving image processing library with many useful methods.
Most common examples of using are presented below:
"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


from optikl.image_processing import represent, image_similarity
print(image_similarity('_92607945_melanoma.jpg', 'photo_2023-11-16_14-54-31.jpg', type_of_encoding='conv', use_features=True))

# print(represent('_92607945_melanoma.jpg'))

