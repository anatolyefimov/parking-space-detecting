import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
import matplotlib.pyplot as plt
from twilio.rest import Client

# Twilio config
twilio_account_sid = 'ACb3faf2906ec37fcbbfd630cbda8e4902'
twilio_auth_token = '8f69357d7a44637cae54aeebe44f620b'
twilio_phone_number = '+16122551706'
destination_phone_number = '+79226260446'
client = Client(twilio_account_sid, twilio_auth_token)
# message = client.messages.create(
#                     body="Отправляйся уже",
#                     from_=twilio_phone_number,
#                     to=destination_phone_number
# )



class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  
    DETECTION_MIN_CONFIDENCE = 0.6

def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


ROOT_DIR = Path(".")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")


COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)


IMAGE_DIR = os.path.join(ROOT_DIR, "images")


IMAGE_SOURCE = "image.png"


model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())


model.load_weights(COCO_MODEL_PATH, by_name=True)

# Location of parking spaces
# <xmin>636</xmin>
# 			<ymin>321</ymin>
# 			<xmax>706</xmax>
# 			<ymax>370</ymax>
parked_car_boxes = parked_car_boxes = np.array([
[321, 636, 370, 706]
])


frame = cv2.imread(IMAGE_SOURCE)
free_space_frames = 0

rgb_image = frame[:, :, ::-1]

# Run the image through the Mask R-CNN model to get results.
results = model.detect([rgb_image], verbose=0)

    # Mask R-CNN assumes we are running detection on multiple images.
    # We only passed in one image to detect, so only grab the first result.
r = results[0]

    # The r variable will now have the results of detection:
    # - r['rois'] are the bounding box of each detected object
    # - r['class_ids'] are the class id (type) of each detected object
    # - r['scores'] are the confidence scores for each detection
    # - r['masks'] are the object masks for each detected object (which gives you the object outline)

    # Filter the results to only grab the car / truck bounding boxes
car_boxes = get_car_boxes(r['rois'], r['class_ids'])

overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes, car_boxes)
free_space = False
for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):

            # For this parking space, find the max amount it was covered by any
            # car that was detected in our image (doesn't really matter which car)
    max_IoU_overlap = np.max(overlap_areas)
    # print(overlap_areas)
            # Get the top-left and bottom-right coordinates of the parking area
    y1, x1, y2, x2 = parking_area

            # Check if the parking space is occupied by seeing if any car overlaps
            # it by more than 0.15 using IoU
    if max_IoU_overlap < 0.15:
                # Parking space not occupied! Draw a green box around it
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        print("etdfgdfg")
                # Flag that we have seen at least one open space
        free_space = True
    else:
                # Parking space is still occupied - draw a red box around it
        cv2.rectangle(frame, (x1, y1), (x2, y2), ( 255, 0,0), 1)

        # Write the IoU measurement inside the box
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))

        # If at least one space was free, start counting frames
        # This is so we don't alert based on one frame of a spot being open.
        # This helps prevent the script triggered on one bad detection.
    if free_space:
        free_space_frames += 1
    else:
            # If no spots are free, reset the count
        free_space_frames = 0

        # If a space has been free for several frames, we are pretty sure it is really free!
if free_space_frames:
            # Write SPACE AVAILABLE!! at the top of the screen
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, f"SPACE AVAILABLE!", (10, 150), font, 3.0, (0, 255, 0), 2, cv2.FILLED)

            # If we haven't sent an SMS yet, sent it!
        #     if not sms_sent:
        #         print("SENDING SMS!!!")
        #         message = client.messages.create(
        #             body="Parking space open - go go go!",
        #             from_=twilio_phone_number,
        #             to=destination_phone_number
        #         )
        #         sms_sent = True

        # # Show the frame of video on the screen
        # cv2.imshow('Video', frame)

print("Cars found in frame of video:")

    # Draw each box on the frame
for box in car_boxes:
    print("Car: ", box)

    y1, x1, y2, x2 = box

        # Draw the box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Show the frame of video on the screen
cv2.imshow('sample image',frame)
cv2.waitKey(0)

# Clean up everything when finished
# video_capture.release()
# cv2.destroyAllWindows()
