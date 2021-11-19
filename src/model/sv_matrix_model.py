import os, sys
import capnp
from os.path import join
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))

from capnp_serial import mapping_capnp,messages_capnp

from abstract_model import UtilityModel
import yaml
from os.path import join

class SVMatrixModel(UtilityModel):
    def __init__(self, model_path):
        with open(join(model_path, "model_config.yaml")) as f:
            self.config_data = yaml.safe_load(f)

        self.colors = []
        self.num_bins = self.config_data["num_bins"]

        for color in self.config_data["colors"]:
            mat_file = color["matrix_file"]
            hue_ranges = color["hue_ranges"]
            range_limits = []
            for hue_range in hue_ranges:
                r = (hue_range["start"], hue_range["end"])
                range_limits.append(r)

            range_limits = sorted(range_limits, key = lambda x : x[0])

            self.colors.append((range_limits, self.read_mat_file(join(model_path, mat_file))))

        self.util_cdf = []
        with open(join(model_path, "util_cdf.txt")) as f:
            for line in f.readlines():
                s = line.split()
                drop_rate = float(s[0])
                util = float(s[1])
                self.util_cdf.append((drop_rate, util))
    
    def read_mat_file(self, mat_file):
        mat = []
        with open(mat_file) as f:
            for line in f.readlines():
                mat.append([])
                s = line.split()
                for x in s:
                    mat[-1].append(float(x))
        return mat

    def match_hue_in_model(self, color_range):
        hue_ranges = []
        for r in color_range.ranges:
            hue_ranges.append((r.hueBegin, r.hueEnd))
        hue_ranges = sorted(hue_ranges, key= lambda x : x[0])

        model_color_idx = 0
        for model_color in self.colors:
            model_range_limits = model_color[0]
            match = True
            if len(hue_ranges) != len(model_range_limits):
                match = False
            else:
                for idx in range(len(model_range_limits)):
                    if model_range_limits[idx][0] != hue_ranges[idx][0] or model_range_limits[idx][1] != hue_ranges[idx][1]:
                        match = False

            if match == True:
                return model_color_idx

            model_color_idx += 1

    def transform_frame_to_model_bin(self, frame_bin, num_frame_bins):
        scale_factor = num_frame_bins/float(self.num_bins)
        return int(frame_bin/scale_factor)

    def compute_utility_for_color(self, model_color, color_hist):
        model_mat = model_color[1]
        frame_mat = [[0 for col in range(self.num_bins)] for row in range(self.num_bins)]
        total = 0
        # Now build a matrix using colorhistogram provided
        row = 0
        for val_bin in color_hist.valueBins:
            col = 0
            for sat in val_bin.counts:
                model_row = self.transform_frame_to_model_bin(row, len(color_hist.valueBins))
                model_col = self.transform_frame_to_model_bin(col, len(val_bin.counts))

                frame_mat[model_row][model_col] += sat

                col += 1
                total += sat
            row += 1

        util = 0
        for row in range(len(frame_mat)):
            for col in range(len(frame_mat[row])):
                if row == 0 or col == 0: # Just double checking not to count
                    continue  
                util += model_mat[row][col]*frame_mat[row][col]/float(total)

        return util

    def get_utility(self, features):
        hist = features.feats[0].feat.hsvHisto
        color_ranges = hist.colorRanges
        color_range_idx = 0
        for color_range in color_ranges:
            model_hue_idx = self.match_hue_in_model(color_range)
            if model_hue_idx != None:
                return self.compute_utility_for_color(self.colors[model_hue_idx], hist.colorHistograms[color_range_idx])

            color_range_idx += 1

    def get_utility_threshold(self, drop_rate):
        for idx in range(len(self.util_cdf)):
            (d, util) = self.util_cdf[idx]
            if d == drop_rate:
                return util

            if d > drop_rate:
                if idx == 0:
                    return util
                else:
                    # Interpolate
                    prev = self.util_cdf[idx-1]
                    slope = (util - prev[1])/(d - prev[0])

                    result = prev[1] + slope*(drop_rate - prev[0])
                    return result

        return 0

if __name__ == "__main__":
    model = SVMatrixModel("/home/surveillance/LoadShedderInterface/src/models/sv_matrix/sv_matrix_red_BINS_8/")
    with open('/home/surveillance/test_feature_request.bin', 'rb') as f:
        util_msg = messages_capnp.UtilityMessage.from_bytes(f.read())
        print (model.get_utility(util_msg.utilityRequest.feats))
