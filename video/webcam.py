
# https://github.com/aiortc/aiortc/tree/master/examples/webcam
# read frames from a webcam and send them to a browser.
# video autostarts
# can be used in an iFrame

# use WebcamVideoStream to run the camera in a thread - reduces wait time for next frame

import argparse
from collections import namedtuple
from threading import Thread
import asyncio
import json
import logging
import os
import platform
import ssl
import datetime

import math

import dlib
import cv2
import numpy
from av import VideoFrame

import paho.mqtt.client as mqtt

from aiohttp import web

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
# from aiortc.contrib.media import MediaPlayer

from pyimagesearch.directioncounter import DirectionCounter
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject


ROOT = os.path.dirname(__file__)


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

# This class originally from 
# https://github.com/aiortc/aiortc/tree/master/examples/videostream-cli
class OutputVideoStreamTrack(VideoStreamTrack):
    """
    A video track that returns camera video.
    """

    def __init__(self):
        super().__init__()  # don't forget this!
        self.counter = 0

        # generate flag
        data_bgr = numpy.hstack(
            [
                self._create_rectangle(
                    width=imageHeigth//3, height=imageHeigth, color=(255, 0, 0)
                ),  # blue
                self._create_rectangle(
                    width=imageHeigth//3+1, height=imageHeigth, color=(255, 255, 255)
                ),  # white
                self._create_rectangle(width=imageHeigth//3, height=imageHeigth, color=(0, 0, 255)),  # red
            ]
        )

        # shrink and center it
        M = numpy.float32([[0.5, 0, imageWidth / 4], [0, 0.5, imageHeigth / 4]])
        data_bgr = cv2.warpAffine(data_bgr, M, (imageWidth, imageHeigth))

        # compute animation
        omega = 2 * math.pi / imageHeigth
        id_x = numpy.tile(numpy.array(range(imageWidth), dtype=numpy.float32), (imageHeigth, 1))
        id_y = numpy.tile(
            numpy.array(range(imageHeigth), dtype=numpy.float32), (imageWidth, 1)
        ).transpose()

        self.frames = []
        for k in range(30):
            phase = 2 * k * math.pi / 30
            map_x = id_x + 10 * numpy.cos(omega * id_x + phase)
            map_y = id_y + 10 * numpy.sin(omega * id_x + phase)
            self.frames.append(
                VideoFrame.from_ndarray(
                    cv2.remap(data_bgr, map_x, map_y, cv2.INTER_LINEAR), format="bgr24"
                )
            )
    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame = self.frames[self.counter % 30]
        frame.pts = pts
        frame.time_base = time_base
        self.counter += 1

        if (trafficDetector.ready):
            # there an image we haven't used yet
            image = trafficDetector.read()
        elif (webcam.grabbed):
            # check if camera has a frame ready
            # get camera frame
            image = webcam.read()
        else:
            #use an blank frame
            image = create_blank(imageWidth, imageHeigth, rgb_color=(88,111,88))

        # perform edge detection
        #image = frame.to_ndarray(format="bgr24")
        #image = cv2.cvtColor(cv2.Canny(image, 100, 200), cv2.COLOR_GRAY2BGR)

        # add date/time
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (5,imageHeigth -5)
        fontScale              = .4
        fontColor              = (255,0,0)
        #fontColor              = (0,0,0)
        lineType               = 1
        cv2.putText(image,str(datetime.datetime.now()), 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)

        #flood the MQTT messaging: test only
        #mqttClient.publish("TrafficCounter/TEST", str(datetime.datetime.now()))

        # rebuild a VideoFrame, preserving timing information
        new_frame = VideoFrame.from_ndarray(image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
        

    def _create_rectangle(self, width, height, color):
        data_bgr = numpy.zeros((height, width, 3), numpy.uint8)
        data_bgr[:, :] = color
        return data_bgr
        
        
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print("ICE connection state is %s" % pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # # open media source
    # if args.play_from:
        # player = MediaPlayer(args.play_from)
    # else:
        # options = {"framerate": "30", "video_size": "640x480"}
        # if platform.system() == "Darwin":
            # player = MediaPlayer("default:none", format="avfoundation", options=options)
        # else:
            # player = MediaPlayer("/dev/video0", format="v4l2", options=options)

    await pc.setRemoteDescription(offer)
    for t in pc.getTransceivers():
        # if t.kind == "audio" and player.audio:
            # pc.addTrack(player.audio)
        # elif t.kind == "video" and player.video:
            # #pc.addTrack(player.video)
            # pc.addTrack(OutputVideoStreamTrack())
        if t.kind == "video":
            pc.addTrack(OutputVideoStreamTrack())

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def on_shutdown(app):
    # halt camera
    webcam.stop()
    
    # halt image processor
    trafficDetector.stop()
    
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

class WebcamVideoStream:
    #https://github.com/jrosebr1/imutils/blob/master/imutils/video/webcamvideostream.py
    def __init__(self, src=0, name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        # https://techoverflow.net/2018/12/18/how-to-set-cv2-videocapture-image-size/
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, imageWidth)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, imageHeigth)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

class detector:
    #https://github.com/jrosebr1/imutils/blob/master/imutils/video/webcamvideostream.py
    # based on the WebcamVideoStream class; which this reads from
    # creates a thread to process current frame looking for 
    # object crossing a line.
    # counts movement, and provides image marked with detected objects
    def __init__(self):

        # initialize the thread name
        self.name = "TrafficCounter"

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        
        #initiallize variable that determines if frame is ready to be consumed
        self.ready = False
        
        #initialze the variable that hods the processed image
        self.frame = ''
        
        #initialize variable for status of the detector
        self.status = "Initializing"
        
        # total number of frames processed
        # used to determine if it's time for a 'deep dive' detection
        self.totalFrames = 0
        
        # instantiate our centroid tracker, then initialize a list to store
        # each of our dlib correlation trackers, followed by a dictionary to
        # map each unique object ID to a trackable object
        self.centroid_Tracker = CentroidTracker(maxDisappeared=20, maxDistance=30)
        self.trackers = []
        self.trackableObjects = {}

        # initialize the direction info variable (used to store information
        # such as up/down or left/right people count)
        self.directionInfo = None

        # number of frames to skip between detections
        self.deepDedectionOnFrame = 30
        
        #a fake system message to make the demo look good
        mqttClient.publish("{}/{}".format(mqtt_client_name,"environment/cpu_temp"), "72C")

    def start(self):
        # start the thread to process frames
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, process next frame
            (self.ready, self.frame) = self.trackMovment()

    def read(self):
        # indicate that image isn't yet ready. avoid pulling an old image
        #self.ready = False
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        
    def trackMovment(self):
        # this function is the entire purpose of this script
        # using,  as a base,  "\HackerBundle_Code\chapter13-object_detection_ncs\people_counter_openvino.py"
        # examine frame for movement to count
        #image = create_blank(imageWidth, imageHeigth, rgb_color=(88,111,88))
        #print("tracking")
        image = webcam.read()

        """
        #convert into a cartoon
        # prepare color
        img_color = cv2.pyrDown(cv2.pyrDown(image))
        for _ in range(6):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        img_color = cv2.pyrUp(cv2.pyrUp(img_color))

        # prepare edges
        img_edges = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img_edges = cv2.adaptiveThreshold(
            cv2.medianBlur(img_edges, 1),
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            2,
        )
        img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

        # combine color and edges
        imageCartoon = cv2.bitwise_and(img_color, img_edges)
        """
        
        # perform edge detection to anonymize the video
        imageOut = cv2.cvtColor(cv2.Canny(image, 200, 300), cv2.COLOR_GRAY2BGR)
        imageOut = cv2.bitwise_not(imageOut)
        #imageOut = image

        # convert the frame from BGR to RGB for dlib
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # initialize the current status along with our list of bounding
        # box rectangles returned by object detector
        self.status = "Waiting"
        rects = []

        # every so often, run a more computationally expensive
        # object detection method to aid our tracker
        if self.totalFrames % self.deepDedectionOnFrame == 0:
            # set the status and initialize our new set of object trackers
            self.status = "Detecting"
            self.trackers = []

            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(image, size=(300, 300), ddepth=cv2.CV_8U)
            net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5, 127.5, 127.5])
            detections = net.forward()

            # loop over the detections
            for i in numpy.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by requiring a minimum
                # confidence
                desiredConfidence = 0.4
                if confidence > desiredConfidence:
                    # extract the index of the class label from the detections list
                    idx = int(detections[0, 0, i, 1])

                    # if the class label is not a person, ignore it
                    if CLASSES[idx] != "person":
                    #print("no persons found")
                        continue
                    else:
                        #print("Person")
                        pass

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * numpy.array(
                        [imageWidth, imageHeigth, imageWidth, imageHeigth])
                    (startX, startY, endX, endY) = box.astype("int")

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    self.trackers.append(tracker)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing
        # throughput
        else:
            # loop over the trackers
            for tracker in self.trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                self.status = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))
                #print(startX, startY, endX, endY)

        # check if the direction is *vertical*
        if directionMode == "vertical":
            # draw a horizontal line in the center of the frame -- once an
            # object crosses this line we will determine whether they were
            # moving 'up' or 'down'
            cv2.line(imageOut, (0, imageWidth // 2), (imageWidth, imageHeigth // 2), (0, 255, 255), 2)

        # otherwise, the direction is *horizontal*
        else:
            # draw a vertical line in the center of the frame -- once an
            # object crosses this line we will determine whether they were
            # moving 'left' or 'right'
            cv2.line(imageOut, (imageWidth // 2, 0), (imageWidth // 2, imageHeigth), (0, 255, 255), 2)

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = self.centroid_Tracker.update(rects)

        # loop over the tracked objects
        info = ''       # a placeholder so text created within this loop can be used later
        for (objectID, centroid) in objects.items():
            # grab the trackable object via its object ID
            tracked_Object = self.trackableObjects.get(objectID, None)
            #print("ObID ", objectID, self.trackableObjects)
            
            if tracked_Object is None:
                # create a new trackable object if needed
                tracked_Object = TrackableObject(objectID, centroid)
                #print("new tracked object")
            else:
                # otherwise, there is a trackable object so we can utilize it
                # to determine direction
                # find the direction and update the list of centroids
                #print("there is a trackable object")
                direction_Counter.find_direction(tracked_Object, centroid)
                tracked_Object.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not tracked_Object.counted:
                    # find the direction of motion of the people
                    self.directionInfo = direction_Counter.count_object(tracked_Object, centroid)

            # store the trackable object in our dictionary
            self.trackableObjects[objectID] = tracked_Object

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            color = (0, 255, 0) if tracked_Object.counted else (0, 0, 255)
            cv2.putText(imageOut, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
            cv2.circle(imageOut, (centroid[0], centroid[1]),6, color, -1)

            # check if there is any direction info available
            if self.directionInfo is not None:
                # construct a list of information as a combination of
                # direction info and status info
                info = self.directionInfo + [("Status", self.status)]
                # send MQTT message
                # for (key, value) in info:
                    # mqttClient.publish("{}/{}".format(mqtt_topic_Detected,key),value)
                # mqttClient.publish("{}/{}".format(mqtt_topic_Detected,"time"),str(datetime.datetime.now()))
            else:
                # otherwise, there is no direction info available yet
                # construct a list of information as status info since we
                # don't have any direction info available yet
                info = [("Status", self.status)]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                #print(i, (k, v))
                text = "{}: {}".format(k, v)
                cv2.putText(imageOut, text, (20, imageHeigth - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # send MQTT message

        # send MQTT message - only if there is something to send
        if (info != ''):
            for (key, value) in info:
                mqttClient.publish("{}/{}".format(mqtt_topic_Detected,key),value)
            mqttClient.publish("{}/{}".format(mqtt_topic_Detected,"time"),str(datetime.datetime.now()))
        
        # increment the total number of frames processed thus far
        self.totalFrames += 1

        return True, imageOut      #imageCartoon

#****************************************************************
#https://stackoverflow.com/questions/9710520/opencv-createimage-function-isnt-working
def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = numpy.zeros((height, width, 3), numpy.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color
    return image
#****************************************************************


def mqtt_on_connect(client, userdata, flags, rc):
    if rc==0:
        client.connected_flag=True    #set flag
        print("MQTT connected OK Returned code=",rc, flush=True)    # the 'flush' empties the stfio buffer so the line appears in the journalctl log https://askubuntu.com/questions/620219/systemd-on-15-04-wont-log-stdout-of-unit
    else:
        client.connected_flag=False    #set flag
        print("Bad connection Returned code=",rc, flush=True)

def mqtt_on_publish(client,userdata,msgID):             #create function for callback
    #print("data published \n")
    pass

def mqtt_on_disconnect(client, userdata, rc):
    logging.info("disconnecting reason  "  +str(rc))
    print("DISCONNECTED Returned code="  +str(rc), flush=True)
    client.connected_flag=False


###############################################################
# start of app
###############################################################

# can only grab images/frames that the camera supports
# list valid formats of a USB camera => v4l2-ctl --list-formats-ext --device=0
# Resolution names: https://en.wikipedia.org/wiki/Display_resolution#/media/File:Vector_Video_Standards8.svg
cameraFormat = namedtuple('cameraFormat', 'width height fps')
cameraXGA = cameraFormat(width=1024, height=600, fps=30)
cameraVGA = cameraFormat(width=640, height=480, fps=30)
cameraQVGA = cameraFormat(width=320, height=240, fps=30)
imageWidth = cameraVGA.width
imageHeigth = cameraVGA.height

#parameters for MQTT
mqtt_broker_address="10.0.3.139"
mqtt_port=1883 
mqtt_client_name = "TrafficCounter"
mqtt_topic_Detected = '{}/TrafficDetected'.format(mqtt_client_name)

# initialize the list of class labels MobileNet SSD detects
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]


# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(
                "mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

# set the preferable target processor to CPU (since Movidious is not installed)
# and preferable backend to OpenCV
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
print("[INFO] model loaded")


pcs = set()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC webcam demo")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)", default="fullchain.pem")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)" , default="privkey.pem")
    parser.add_argument("--play-from", help="Read the media from a file and sent it."),
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8888, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    #initialize the MQTT Client
    # http://www.steves-internet-guide.com/client-objects-python-mqtt/
    mqtt.Client.connected_flag=False    #create flag in class
    mqttClient = mqtt.Client(mqtt_client_name)    #create new instance
    mqttClient.on_connect=mqtt_on_connect    #bind call back function
    mqttClient.on_publish = mqtt_on_publish    #assign function to callback
    mqttClient.on_disconnect=mqtt_on_disconnect    #bind call back function
    mqttClient.connect(mqtt_broker_address,mqtt_port)    #establish connection
    mqttClient.loop_start() #start the loop

    # initialize camera; start grabbing frames
    webcam = WebcamVideoStream().start()

    # initialize the 
    trafficDetector = detector().start()

    # instantiate our direction counter
    directionMode = "horizontal"   # "vertical"
    direction_Counter = DirectionCounter(directionMode, imageHeigth, imageWidth)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)
