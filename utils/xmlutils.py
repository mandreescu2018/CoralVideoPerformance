
import xml.etree.ElementTree as ET
import os

xmlpath = '../video/Walk-train/Annotations/SJTU-SEIEE-94_Walk_0050/SJTU-SEIEE-94_Walk_0050-0007.xml'


class GroundTruthBoundingBox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self._label = None
        self.id = 0

    def __repr__(self):
        return f'BBox(xmin = {self.xmin}, ymin = {self.ymin}, ' \
               f'xmax = {self.xmax}, ymax = {self.ymax})'


class GroundTruthAnnotation:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.bbox = GroundTruthBoundingBox(xmin, ymin, xmax, ymax)
        self.id = 0
        self.score = 1

class GroundTruth:
    def __init__(self, file_xml):

        self.objects = []

        if os.path.exists(file_xml):
            tree = ET.parse(file_xml)
            root = tree.getroot()
            self.node = root.find('pedestriandescription/bndbox')
            self.parse_xml()

    def parse_xml(self):
        box = self.node.find('item')
        if box is None:
            lst = []
            for child in self.node:
                lst.append(child.text)
            lst = [int(item) for item in lst]
            bbox = GroundTruthAnnotation(lst[0], lst[1], lst[2], lst[3])

            self.objects.append(bbox)
        else:
            for bbox in self.node.findall('item'):
                lst = []
                for child in bbox:
                    lst.append(child.text)

                lst = [int(item) for item in lst]
                bbox = GroundTruthAnnotation(lst[0], lst[1], lst[2], lst[3])
                # bbox.Label = obj['label']
                bbox.Label = 'Person'
                self.objects.append(bbox)





if __name__ == '__main__':
    ann = GroundTruth(xmlpath)
    for gb in ann.objects:
        print(gb)