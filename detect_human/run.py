import sys
import yaml
from scake import Scake

from hog_feature import HOGFeature
from object_detector import ObjectDetector
from nms import NMS


def main(yaml_path):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    s = Scake(config, class_mapping=globals())
    s.run()
    pass


if __name__ == "__main__":
    main(yaml_path=r"hog.yaml")
