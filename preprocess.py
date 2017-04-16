import cv2

def preprocess(image):
    yhigh = 50 # y-coordinates from the top
    ydown = 155 # y-coordinates from the top
    resize = image[yhigh:ydown,0:319]
    resize = cv2.resize(resize,(204,70))
    return resize