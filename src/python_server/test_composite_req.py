import os
import sys
import capnp
script_dir = os.path.dirname(os.path.realpath(__file__))                                                        
sys.path.insert(0, os.path.join(script_dir, "../"))                                                             
import mapping_features
from capnp_serial import mapping_capnp, messages_capnp
f = open(sys.argv[1], 'rb')                                                                                   
utility = messages_capnp.UtilityMessage.read(f)                                                                 
assert utility.messageType == messages_capnp.UtilityMessage.Type.utilityRequest
request = utility.utilityRequest
assert request.mode == "max_cdf"
features = request.feats
assert len(features.feats) == 1
feature = features.feats[0]                                                                                     
assert feature.type == mapping_capnp.Feature.Type.hsvHistogram
hsvhisto = feature.feat.hsvHisto
colorHistos = hsvhisto.colorHistograms
colorRanges = hsvhisto.colorRanges
binSize = hsvhisto.binSize
pixels = hsvhisto.totalCountedPixels
assert len(colorHistos) == 2
assert len(colorRanges) == 2
assert binSize == 16
assert pixels > 0
assert len(colorHistos[0].valueBins) == 16
for valueBin in colorHistos[0].valueBins:
    assert len(valueBin.counts) == 16
assert len(colorHistos[1].valueBins) == 16
for valueBin in colorHistos[1].valueBins:
    assert len(valueBin.counts) == 16
assert len(colorRanges[0].ranges) == 1
assert len(colorRanges[1].ranges) == 2
