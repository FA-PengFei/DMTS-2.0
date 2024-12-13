import sys
import traceback
import tellopy
from pynput import keyboard
import av
import cv2.cv2 as cv2  # for avoidance of pylint error
import numpy
import time
import pathlib

def main():
    drone = tellopy.Tello()

    speed = 25

    # Tracking stuff
    labelsPath = "{}/obj.names".format(pathlib.Path().absolute())
    confidence1 = 0.5
    threshold = 0.3
    configFile = "{}/yolov3-tiny.cfg".format(pathlib.Path().absolute())
    weights = "{}/yolov3-tiny_21000.weights".format(pathlib.Path().absolute())
    LABELS = open(labelsPath).read().strip().split("\n")
    net = cv2.dnn.readNetFromDarknet(configFile, weights)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    (W, H) = (None, None)
    frame_centerX, frame_centerY = None, None
    mal_drone_xoffset, mal_drone_yoffset = 0, 0

    numpy.random.seed(42)
    COLORS = numpy.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

    try:
        drone.connect()
        # Block until connection is established
        drone.wait_for_connection(60.0)

        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')

        frame_skip = 200
        while True:
            # Sets decoding as video
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                
                start_time = time.time()
                image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                

                # if the frame dimensions are empty, grab them
                if W is None or H is None:
                    (H, W) = image.shape[:2]
                    frame_centerX = int(W/2)
                    frame_centerY = int(H/2)

                # construct a blob from the input frame and then perform a forward
                # pass of the YOLO object detector, giving us our bounding boxes
                # and associated probabilities
                blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                    swapRB=True, crop=False)
                net.setInput(blob)
                start = time.time()
                layerOutputs = net.forward(ln)
                end = time.time()

                # initialize our lists of detected bounding boxes, confidences,
                # and class IDs, respectively
                boxes = []
                confidences = []
                classIDs = []

                # loop over each of the layer outputs
                for output in layerOutputs:
                    # loop over each of the detections
                    for detection in output:
                        # extract the class ID and confidence (i.e., probability)
                        # of the current object detection
                        scores = detection[5:]
                        classID = numpy.argmax(scores)
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > confidence1:
                            # scale the bounding box coordinates back relative to
                            # the size of the image, keeping in mind that YOLO
                            # actually returns the center (x, y)-coordinates of
                            # the bounding box followed by the boxes' width and
                            # height
                            box = detection[0:4] * numpy.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top
                            # and and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates,
                            # confidences, and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)

                # apply non-maxima suppression to suppress weak, overlapping
                # bounding boxes
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence1,
                    threshold)

                # ensure at least one detection exists
                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        # draw a bounding box rectangle and label on the frame
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                            confidences[i])
                        cv2.putText(image, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # calculate malicious drone offset
                        # draw arrow to visualise direction
                        mal_drone_xoffset = int(centerX - frame_centerX)
                        mal_drone_yoffset = int(frame_centerY - centerY)
                        cv2.arrowedLine(image, (frame_centerX, frame_centerY), (frame_centerX + mal_drone_xoffset, frame_centerY - mal_drone_yoffset), (0, 0, 255), 5)
                        distance = 100
                        print("Mal_Drone_Xoffset = {0}, Mal_Drone_Yoffset = {1}".format(mal_drone_xoffset, mal_drone_yoffset))
                        if mal_drone_xoffset < -distance:
                            drone.left(speed)
                            time.sleep(1)
                            drone.left(0)
                            break
                        elif mal_drone_xoffset > distance:
                            drone.right(speed)
                            time.sleep(1)
                            drone.right(0)
                            break
                        elif mal_drone_yoffset < -distance:
                            drone.down(speed)
                            time.sleep(1)
                            drone.down(0)
                            break
                        elif mal_drone_yoffset > distance:
                            drone.up(speed)
                            time.sleep(1)
                            drone.up(0)
                            break
                        else:
                            continue
                        
                cv2.imshow('Guardian 1 View', image)
                key = cv2.waitKey(1) & 0xff
                if key == 27:
                    drone.land()
                    drone.quit()
                    cv2.destroyAllWindows()
                    exit()
                    break
                elif key == 9:
                    drone.takeoff()
                elif key == ord('w'):
                    drone.forward(speed)
                    time.sleep(1)
                    drone.forward(0)
                elif key == ord('s'):
                    drone.backward(speed)
                    time.sleep(1)
                    drone.backward(0)
                elif key == ord('a'):
                    drone.left(speed)
                    time.sleep(1)
                    drone.left(0)
                elif key == ord('d'):
                    drone.right(speed)
                    time.sleep(1)
                    drone.right(0)
                elif key == ord('e'):
                    drone.clockwise(speed)
                    time.sleep(1)
                    drone.clockwise(0)
                elif key == ord('q'):
                    drone.counter_clockwise(speed)
                    time.sleep(1)
                    drone.counter_clockwise(0)
                elif key == ord('r'):
                    drone.up(speed)
                    time.sleep(1)
                    drone.up(0)
                elif key == ord('f'):
                    drone.down(speed)
                    time.sleep(1)
                    drone.down(0)
                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)
                    

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()