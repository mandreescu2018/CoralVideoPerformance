import os
import glob
import abc
import xml.etree.ElementTree as ET
import json
from config import settings
from utils.boundingboxutils import BoundingBoxItem

settings.init()

def get_data(images_path, gt_path):
    all_img_files_list = []
    all_json_files_list = []

    for every_folder in os.walk(images_path):
        temp_list = [f for f in glob.glob(os.path.join(every_folder[0], '*.png'))]
        temp_list.sort()
        all_img_files_list += temp_list
    for every_folder in os.walk(gt_path):
        temp_list = [f for f in glob.glob(os.path.join(every_folder[0], '*.json'))]
        temp_list.sort()
        all_json_files_list += temp_list

    # print(len(all_img_files_list))
    # print(len(all_json_files_list))
    return all_img_files_list, all_json_files_list

class GroundTruth:
    def __init__(self, path):
        self.path = path
        self.gt_items = []

class GroundTruthXml(GroundTruth):
    def __init__(self, path, root_str_path = 'pedestriandescription/bndbox'):
        super().__init__(path)

        if os.path.exists(path):
            tree = ET.parse(path)
            root = tree.getroot()
            self.node = root.find(root_str_path)
            self.parse_xml()

    def parse_xml(self):
        box = self.node.find('item')

        if box is None:
            xml_items = [self.node]
        else:
            xml_items = self.node.findall('item')

        for box in xml_items:
            lst = []
            for child in box:
                lst.append(child.text)
            lst = [int(item) for item in lst]
            bbox = BoundingBoxItem(*lst[0:4])
            self.gt_items.append(bbox)


class GroundTruthTxt(GroundTruth):
    def __init__(self, path):
        super().__init__(path)
        self.read_file()


    def read_file(self):
        """
        1  track_id. All rows with the same ID belong to the same path.
        2   xmin. The top left x-coordinate of the bounding box.
        3   ymin. The top left y-coordinate of the bounding box.
        4   xmax. The bottom right x-coordinate of the bounding box.
        5   ymax. The bottom right y-coordinate of the bounding box.
        6   frame_number. The frame that this annotation represents.
        7   lost. If 1, the annotation is outside of the view screen.
        8   occluded. If 1, the annotation is occluded.
        9   generated. If 1, the annotation was automatically interpolated.
        10  label. human, car/vehicle, bicycle.
        :return:
        """
        with open(self.path) as fl:
            for line in fl:
                lst = line.split(' ')
                self.gt_items.append(BoundingBoxItem(*lst[1:5], frame_no=lst[5], subject_out_of_frame=lst[6]))

class GroundTruthJSON(GroundTruth):
    def __init__(self, path):
        super().__init__(path)
        with open(self.path, 'r') as j:
            json_data = json.load(j)

        self.json_data = json_data
        self.parse_json()

    def parse_json(self):
        for obj in self.json_data['objects']:
            if obj['label'] not in ['pedestrian', 'rider', 'sitting person']:
                continue
            lst = obj['bbox']
            # self.xmin = xmin
            # self.ymin = ymin
            # self.xmax = self.xmin + xoffset
            # self.ymax = self.ymin + yoffset

            lst[2] = lst[0] + lst[2]
            lst[3] = lst[1] + lst[3]
            bbox = BoundingBoxItem(*lst[0:4])
            bbox.Label = obj['label']
            self.gt_items.append(bbox)
