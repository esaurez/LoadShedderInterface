import os
import sys
import capnp
import numpy as np
import cv2
import base64 
script_dir=os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))
from capnp_serial import messages_capnp
from capnp_serial import mapping_capnp

import server

PORT = 5556

def test_utility_threshold_function():
    drop_ratio = 0.8

    interface = server.InterfaceReqHandler(PORT)
    util_msg = messages_capnp.UtilityMessage.new_message()
    util_msg.messageType = "utilityThresholdRequest"
    util_msg.init("utilityThresholdRequest")
    util_msg.utilityThresholdRequest.dropRatio = drop_ratio

    print ("Requesting utility threshold for drop ratio = ", drop_ratio)
    reply = interface.handle_msg(util_msg)
    print ("Util threshold returned = ", reply.utilityThresholdResponse.threshold)

def serialize_image(image, format_type):
    _, encoded = cv2.imencode(format_type, image)
    jpg_as_text = base64.b64encode(encoded)
    return jpg_as_text

def test_utility_function():
    encoding_format='.jpg'
    # Create an artificial image
    mat = np.zeros((10,10,3), dtype=np.uint8)
    cv2.randn(mat, 128, 50)

    interface = server.InterfaceReqHandler(PORT)
    util_msg = messages_capnp.UtilityMessage.new_message()
    util_msg.messageType = "utilityRequest"
    util_msg.init("utilityRequest")
    util_msg.utilityRequest.init("feats")
    util_msg.utilityRequest.feats.init("feats", 2)
    util_msg.utilityRequest.feats.feats[0].type = 'foreground'
    util_msg.utilityRequest.feats.feats[0].feat.init('foreground')
    util_msg.utilityRequest.feats.feats[0].feat.foreground.mask.matBinary = serialize_image(mat, encoding_format)
    util_msg.utilityRequest.feats.feats[0].feat.foreground.mask.extension = encoding_format
    util_msg.utilityRequest.feats.feats[1].type = 'foreground'
    util_msg.utilityRequest.feats.feats[1].feat.init('detections',2)
    util_msg.utilityRequest.feats.feats[1].feat.detections[0].label="dog"
    util_msg.utilityRequest.feats.feats[1].feat.detections[0].confidence=0.5
    util_msg.utilityRequest.feats.feats[1].feat.detections[0].left=0
    util_msg.utilityRequest.feats.feats[1].feat.detections[0].right=256
    util_msg.utilityRequest.feats.feats[1].feat.detections[0].top=0
    util_msg.utilityRequest.feats.feats[1].feat.detections[0].bottom=256
    util_msg.utilityRequest.feats.feats[1].feat.detections[1].label="car"
    util_msg.utilityRequest.feats.feats[1].feat.detections[1].confidence=0.8
    util_msg.utilityRequest.feats.feats[1].feat.detections[1].left=256
    util_msg.utilityRequest.feats.feats[1].feat.detections[1].right=512
    util_msg.utilityRequest.feats.feats[1].feat.detections[1].top=0
    util_msg.utilityRequest.feats.feats[1].feat.detections[1].bottom=256

    reply = interface.handle_msg(util_msg)
    print ("Util returned = ", reply.utilityResponse.utility)

if __name__ == "__main__":
    print ("=========================")
    test_utility_threshold_function()
    
    print ("=========================")
    test_utility_function()
