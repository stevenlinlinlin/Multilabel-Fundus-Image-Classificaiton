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
    # Method 1: Using contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    result = cv2.bitwise_and(image, image, mask=mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)
    
    # Method 2: Using intensity profile
    # # Step 1: Convert to red channel and find image dimensions
    # red_channel = image[:, :, 2]  # Assuming the image is in BGR format
    # h, w = red_channel.shape

    # # Step 2: Calculate center lines
    # Hcenterline = h // 2
    # Vcenterline = w // 2

    # # Step 3: Draw scanning lines and calculate intensity profile
    # horizontal_line = red_channel[Hcenterline, :]
    # vertical_line = red_channel[:, Vcenterline]

    # # Step 4: Calculate threshold using empirical factor
    # th = max(horizontal_line) * 0.06

    # # Find transitions based on threshold
    # horizontal_transitions = np.where(np.diff(horizontal_line > th))[0]
    # vertical_transitions = np.where(np.diff(vertical_line > th))[0]

    # # Ensure there are at least two transitions to form a rectangle
    # if len(horizontal_transitions) >= 2 and len(vertical_transitions) >= 2:
    #     X1, X2 = horizontal_transitions[[0, -1]]
    #     Y1, Y2 = vertical_transitions[[0, -1]]

    #     # Step 5: Crop the FOV based on found coordinates
    #     fov_image = image[Y1:Y2, X1:X2]
    #     return Image.fromarray(fov_image)
    # else:
    #     print("No valid transitions found")
    #     return None  # In case no valid transitions are found
    
def enhance_image(image, r, eps, enhancement_factor):
    image = image.astype(np.float32)
    # Apply guided filter to smooth the image
    smoothed = cv2.ximgproc.guidedFilter(guide=image, src=image, radius=r, eps=eps, dDepth=-1)
    # Calculate the detail image
    detail = image - smoothed
    # Enhance the image
    enhanced = image + enhancement_factor * detail
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced)

def image_loader(path,transform):
    try:
        image = cv2.imread(path)
    except FileNotFoundError: # weird issues with loading images on our servers
        # print('FILE NOT FOUND')
        time.sleep(10)
        image = cv2.imread(path)

    # image = image.convert('RGB')
    # image = fov_extractor(np.array(image))
    image = enhance_image(image, 5, 0.01 * 255 * 255, 5)
    
    if transform is not None:
        image = transform(image)

    return image