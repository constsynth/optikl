"""
Hello! This is constantly evolving image processing library with many useful methods.
Most common examples of using are presented below:
"""

from optikl.image_processing import find_face, crop_center, represent, super_resolution, image_similarity
print(image_similarity('im2.jpg', 'im1.jpg', type_of_encoding='conv'))
