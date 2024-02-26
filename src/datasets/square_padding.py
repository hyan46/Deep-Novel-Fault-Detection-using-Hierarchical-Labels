""" 
Taken from https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/4 
to be able to combat with varying input sizes in the Rolling dataset. Pads 
the minimum of height and width with 0 so that the image becomes square. Once 
all images are square, they can be resized to any NxN without losing 
proportions.

Usage: Just make an instance and add it to your transforms.Compose object
before Resize().
"""
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

import numpy as np


# class SquarePad:
#     def __call__(self, image):
#         w, h = image.size
#         max_wh = np.max([w, h])
#         hp = int((max_wh - w) / 2)
#         vp = int((max_wh - h) / 2)
#         padding = (hp, vp, hp, vp)
#         return F.pad(image, padding, 0, "constant")
    


class SquarePad:
    def __call__(self, image,size = 80):
        transform = transforms.CenterCrop(size)
        return transform(image)


# class SquarePad:
#     def __call__(self, image,size = 80):
#         w, h = image.size
#         print(type(image))
#         (d_w,d_h) = int((w-size)/2), int((h-size)/2)

#         if d_w >0:
#             image = image[d_w:(w - d_w),:]
#             wp = 0   
#         else: 
#             wp = -1*d_w


#         if d_h >0:
#             image = image[:,d_h:h-d_h]
#             hp = 0  #hp:  padding of height
#         else:
#             hp = -1*d_h

#         padding = (hp, wp, hp, wp)
        
#         return F.pad(image, padding, 0, "constant")
