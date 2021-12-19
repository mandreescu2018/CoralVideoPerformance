
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


class BoundingBoxItem:
    def __init__(self, xmin, ymin, xmax, ymax, score=0, frame_no=0, subject_out_of_frame=0):
        self.bbox = BBox(xmin, ymin, xmax, ymax)
        self.frame_no = int(frame_no)
        self.label = ''
        self.score = score
        self.out_of_frame = int(subject_out_of_frame)

    def to_list(self):
        return [self.bbox.xmin, self.bbox.ymin, self.bbox.xmax, self.bbox.ymax]

    def __repr__(self):
        return str(self.bbox) + f" frame: {self.frame_no} \n"
