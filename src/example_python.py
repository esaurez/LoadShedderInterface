#!/usr/bin/env python3

import base64
import capnp
import cv2
import numpy as np

from capnp_serial import mapping_capnp

def serialize_image(image, format_type):
    _, encoded = cv2.imencode(format_type, image)
    jpg_as_text = base64.b64encode(encoded)
    return jpg_as_text

def deserialize_image(base64encoded_frame):
    encoded_image = base64.b64decode(base64encoded_frame)
    as_np = np.frombuffer(encoded_image, dtype=np.uint8)
    return cv2.imdecode(as_np, flags=cv2.IMREAD_ANYCOLOR)

def initialize_and_serialize_object():
    encoding_format='.jpg'
    # Create an artificial image
    mat = np.zeros((10,10,3), dtype=np.uint8)
    cv2.randn(mat, 128, 50)

    # Initialize struct
    features = mapping_capnp.Features.new_message()
    features.init('feats',2)
    features.feats[0].type = 'foreground'
    features.feats[0].feat.init('foreground')
    features.feats[0].feat.foreground.mask.matBinary = serialize_image(mat, encoding_format)
    features.feats[0].feat.foreground.mask.extension = encoding_format 
    features.feats[1].type = 'foreground'
    features.feats[1].feat.init('detections',2)
    features.feats[1].feat.detections[0].label="dog"
    features.feats[1].feat.detections[0].confidence=0.5
    features.feats[1].feat.detections[0].left=0
    features.feats[1].feat.detections[0].right=256
    features.feats[1].feat.detections[0].top=0
    features.feats[1].feat.detections[0].bottom=256
    features.feats[1].feat.detections[1].label="car"
    features.feats[1].feat.detections[1].confidence=0.8
    features.feats[1].feat.detections[1].left=256
    features.feats[1].feat.detections[1].right=512
    features.feats[1].feat.detections[1].top=0
    features.feats[1].feat.detections[1].bottom=256
    
    # Return serialized bytes
    return features.to_bytes()

    # Save to file
    # f = open('example.bin', 'w+b')
    # features.write(f)

def deserialize_message(serialized):
    features = mapping_capnp.Features.from_bytes(serialized)
    print("This should print dog: {}".format(features.feats[1].feat.detections[0].label))
    print("This should print car: {}".format(features.feats[1].feat.detections[1].label))
    recovered_mask = deserialize_image(features.feats[0].feat.foreground.mask.matBinary)

if __name__ == "__main__":
    serialized_features = initialize_and_serialize_object()
    deserialize_message(serialized_features)
