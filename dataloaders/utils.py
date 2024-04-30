from PIL import Image
import numpy as np
import random
import time
import hashlib
import cv2

def get_unk_mask_indices(image,testing,num_labels,known_labels):
    if testing:
        # for consistency across epochs and experiments, seed using hashed image array 
        random.seed(hashlib.sha1(np.array(image)).hexdigest())
        unk_mask_indices = random.sample(range(num_labels), (num_labels-int(known_labels)))
    else:
        # sample random number of known labels during training
        if known_labels>0:
            random.seed()
            num_known = random.randint(0,int(num_labels*0.75))
        else:
            num_known = 0

        unk_mask_indices = random.sample(range(num_labels), (num_labels-num_known))

    return unk_mask_indices

def fov_extractor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    result = cv2.bitwise_and(image, image, mask=mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)

def image_loader(path,transform):
    try:
        image = Image.open(path)
    except FileNotFoundError: # weird issues with loading images on our servers
        # print('FILE NOT FOUND')
        time.sleep(10)
        image = Image.open(path)

    image = image.convert('RGB')
    image = fov_extractor(np.array(image))
    
    if transform is not None:
        image = transform(image)

    return image