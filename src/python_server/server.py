import zmq
import time
import sys
import os
import capnp

script_dir=os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))

from capnp_serial import mapping_capnp

port = "5556"
if len(sys.argv) > 1:
    port =  sys.argv[1]

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)

while True:
    #  Wait for next request from client
    message = socket.recv()
    point  = mapping_capnp.Point.from_bytes(message)
    print ("Received request: ", point.x)

    pointReply = mapping_capnp.Point.new_message()
    pointReply.x = 1336
    pointReply.y = 100
    #point.set_x(5)
    socket.send(pointReply.to_bytes())


