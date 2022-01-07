import yaml
import sys

infile=sys.argv[1]
key=sys.argv[2]

with open(infile) as f:
    data = yaml.safe_load(f)
    print (data[key])
