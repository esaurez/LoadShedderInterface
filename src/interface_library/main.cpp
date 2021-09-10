#include <iostream>
#include "comm_agent.h"

void addFullColorHistogram(Feature::Builder feature) {
    feature.setType(Feature::Type::FULL_COLOR_HISTOGRAM);
    auto unionFeat = feature.initFeat();
    auto fullHisto = unionFeat.initWholeHisto();
    auto histo = fullHisto.initHisto();
    
    // Now initializing the histogram
    auto channels = histo.initChannels(3);
    auto histSize = histo.initHistSize(3);
    auto dimensions = histo.initDimensions(3);
    auto ranges = histo.initRanges(3);
    int totalPixels = 256;
    histo.setTotalCountedPixels(static_cast<long unsigned int>(totalPixels));
    for(unsigned int chanIdx = 0; chanIdx < 3; chanIdx++)
    {
        channels.set(chanIdx, static_cast<unsigned char>(chanIdx));
        histSize.set(chanIdx, static_cast<unsigned char>(256));
        dimensions.set(chanIdx, 1);
        ranges[chanIdx].setLowerLim(0);
        ranges[chanIdx].setUpperLim(256);
    }
    auto counts = histo.initCounts(3);
    auto bCount = counts[0].initCount(256);
    auto gCount = counts[1].initCount(256);
    auto rCount = counts[2].initCount(256);
    for(unsigned int histIdx = 0; histIdx < 256; histIdx++)
    {
        bCount.set(histIdx,1);
        gCount.set(histIdx,1);
        rCount.set(histIdx,1);
    }    
}

int main() {
    std::string mode = "max_cdf";
    CommAgent agent("tcp://localhost:5556");
    std::cout << agent.getUtilityThreshold(0, mode) << std::endl; 

    ::capnp::MallocMessageBuilder message;
    Features::Builder features = message.initRoot<Features>();
    auto feats = features.initFeats(1);
    addFullColorHistogram(feats[0]);
    std::cout << agent.getUtilityValue(features, mode) << std::endl;
}
