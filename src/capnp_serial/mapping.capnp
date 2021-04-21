@0xa884193775a6279a;

struct SerializedMat{
  matBinary @0 :Data;
  extension @1 :Text;
}

struct Foreground{
  mask @0 :SerializedMat;
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

struct Histogram{
  counts @0 :List(UInt64);
  histSize @1 :List(UInt8);
  ranges @2 :List(Range);
  channels @3 :List(UInt8);
  dimensions @4 :UInt8;
}

struct FullHistogram{
  histo @0 :Histogram;
} 

struct BlobHistogram{
  histo @0 :Histogram;
  contour @1 :Contour;
}

struct Detection{
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
  }
  type @0 :Type;
  feat :union {
    foreground @1 :Foreground;
    contours @2 :List(Contour);
    wholeHisto @3 :FullHistogram;
    blobHisto @4 :List(BlobHistogram);
    detections @5 :List(Detection);
  }
}

struct Features{
  feats @0: List(Feature);
}

struct LabeledData{
  label @0 :Bool;
  feats @1 :Features;
}

struct Training{
  data @0 :List(LabeledData);
}
