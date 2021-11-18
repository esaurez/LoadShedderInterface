@0xa884193775a6279a;

struct FeatSerializedMat{
  matBinary @0 :Data;
  extension @1 :Text;
}

struct Foreground{
  mask @0 :FeatSerializedMat;
}

struct Point{
  x @0 :UInt16;
  y @1 :UInt16;
}

struct Contour{
  points @0 :List(Point); 
}

struct Range{
  lowerLim @0 :UInt64; 
  upperLim @1 :UInt64; 
}

struct Count{
 count @0 :List(UInt64);
}

struct Histogram{
  counts @0 :List(Count);
  histSize @1 :List(UInt8);
  ranges @2 :List(Range);
  channels @3 :List(UInt8);
  dimensions @4 :List(UInt8);
  totalCountedPixels @5 :UInt64;
}

struct FullHistogram{
  histo @0 :Histogram;
} 

struct SaturationHistogram
{
  counts @0 :List(UInt64);
}

struct ValueHistogram
{
  valueBins @0 :List(SaturationHistogram);
}

struct CRange{
  hueBegin @0 :UInt8;
  hueEnd @1 :UInt8;
}

struct ColorDescription{
  ranges @0 :List(CRange);
}

struct HsvHistogram{
  colorHistograms @0 :List(ValueHistogram);
  binSize @1 :UInt8;
  totalCountedPixels @2 :UInt64;
  colorRanges @3 :List(ColorDescription);
} 

struct BlobHistogram{
  histo @0 :Histogram;
  contour @1 :Contour;
}

struct FeatDetection{
  label @0 :Text;
  confidence @1 :Float64;
  left @2 :UInt64;
  top @3 :UInt64;
  right @4 :UInt64;
  bottom @5 :UInt64;
}

struct Feature{
  enum Type {
    foreground @0;
    contours @1;
    fullColorHistogram @2;
    perBlobColorHistogram @3;
    detections @4;
    hsvHistogram @5;
  }
  type @0 :Type;
  feat :union {
    foreground @1 :Foreground;
    contours @2 :List(Contour);
    wholeHisto @3 :FullHistogram;
    blobHisto @4 :List(BlobHistogram);
    detections @5 :List(FeatDetection);
    hsvHisto @6 :HsvHistogram;
  }
}

struct Features{
  feats @0: List(Feature);
}

struct DetectionMetadata{
  minSize @0 :UInt32;
  maxSize @1 :UInt32;
  totalDetectionSize @2 :UInt32;
  totalDetections @3 :UInt32;
}

struct LabeledData{
  frameIndex @0 :Text;
  label @1 :Bool;
  feats @2 :Features;
  detections @3 :DetectionMetadata;
}

struct VideoFeatures {
  videoDescription @0 :Text;
  data @1 :List(LabeledData);
}

struct Training{
  data @0 :List(VideoFeatures);
}
