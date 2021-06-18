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

struct UtilityThresholdRequest {
    dropRatio @0 :Float64;
}

struct UtilityThresholdResponse {
    threshold @0 :Float64;
}

struct UtilityRequest {
    feats @0 :Mapping.Features;
}

struct UtilityResponse {
    utility @0 :Float64;
}
