#!/usr/bin/env python3

import capnp
import capnp_serial.mapping_capnp

def initialize_object():
    message = mapping_capnp.RockletUpdate.new_message()
