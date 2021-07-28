import os
import abc
import xml.etree.ElementTree as ET


class BBox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)

    def __str__(self):
        return f"x min: {self.xmin}, y min: {self.ymin}, x max: {self.xmax}, y max: {self.ymax}"

    def coordinates_list(self):
        return list((self.xmin, self.ymin, self.xmax, self.ymax))


class GroundTruthItem:
    def __init__(self, xmin, ymin, xmax, ymax, score, frame_no = 0):
        self.bbox = BBox(xmin, ymin, xmax, ymax)
        self.frame_no = int(frame_no)
        self.label = ''
        self.score = score

    def to_list(self):
        return [self.bbox.xmin, self.bbox.ymin, self.bbox.xmax, self.bbox.ymax]

    def __repr__(self):
        return str(self.bbox) + f" frame: {self.frame_no} \n"


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
            bbox = GroundTruthItem(*lst[0:4])
            self.gt_items.append(bbox)


class GroundTruthTxt(GroundTruth):
    def __init__(self, path):
        super().__init__(path)


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
                self.gt_items.append(GroundTruthItem(*lst[1:6]))

class GroundTruthJSON(GroundTruth):
    def __init__(self, path):
        super().__init__(path)
