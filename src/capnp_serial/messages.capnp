@0xda5899221e2300af;

using Mapping = import "mapping.capnp";

struct UtilityMessage {
    messageType @0 :Type;

    utilityThresholdRequest @1 :UtilityThresholdRequest;
    utilityThresholdResponse @2 :UtilityThresholdResponse;
    utilityRequest @3 :UtilityRequest;
    utilityResponse @4 :UtilityResponse;

    enum Type {
        utilityThresholdRequest @0;
        utilityThresholdResponse @1;
        utilityRequest @2;
        utilityResponse @3;
    }
}

struct VideoIdxDropRatio {
    videoIdx @0 :UInt64;
    dropRatio @1 :Float64;
}

struct VideoIdxUtilityThreshold {
    videoIdx @0 :UInt64;
    threshold @1 :Float64;
}

struct UtilityThresholdRequest {
    dropRatios @0 :List(VideoIdxDropRatio);
    mode @1 :Text;  
}

struct UtilityThresholdResponse {
    thresholds @0 :List(VideoIdxUtilityThreshold);
}

struct UtilityRequest {
    feats @0 :Mapping.Features;
    mode @1 :Text;  
    videoIdx @2 :UInt64;
}

struct UtilityResponse {
    utility @0 :Float64;
}
