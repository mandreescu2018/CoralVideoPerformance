import os
import glob
from pathlib import Path
from utils import groundtruthutils

class DataUtils:
    def __init__(self, configuration):
        self.configuration = configuration
    # result = eval(function_name + "()")

    def get_video_files(self):
        function_name = "get_video_files_" + self.configuration.section.lower()
        result = eval("self." + function_name + "()")
        return result

    def get_video_files_shad(self):

        files = os.listdir(self.configuration.data_path)
        video_files_dict = {i: files[i].rsplit('.', 1)[0] for i in range(0, len(files))}
        return video_files_dict

    def get_video_files_dict(self):
        files = [os.path.abspath(os.path.join(self.configuration.data_path, p)) for p in os.listdir(self.configuration.data_path)]
        annotations = [os.path.abspath(os.path.join(self.configuration.annotations_path, p)) for p in
                       os.listdir(self.configuration.annotations_path)]
        video_files_dict = [{'data': files[i], 'annotation': annotations[i]} for i in range(0, len(files))]
        return video_files_dict

    def get_data(self):
        images_path = os.path.join(self.configuration.home_path, self.configuration.data_path)
        annotations_path = os.path.join(self.configuration.home_path, self.configuration.annotations_path)

        all_img_files_list = []
        all_json_files_list = []

        for every_folder in os.walk(images_path):
            temp_list = [f for f in glob.glob(os.path.join(every_folder[0], '*.png'))]
            temp_list.sort()
            all_img_files_list += temp_list
        for every_folder in os.walk(annotations_path):
            temp_list = [f for f in glob.glob(os.path.join(every_folder[0], '*.json'))]
            temp_list.sort()
            all_json_files_list += temp_list

        return all_img_files_list, all_json_files_list


    def get_annotation_by_video_frame(self, xmlPath, frame_number):

        xmlpath = Path(xmlPath)

        file_name = os.path.basename(xmlPath).split('.')[0] + '-{}.xml'
        f_name = file_name.format(str(int(frame_number)).zfill(4))
        file_name_for_metrics = f_name.split(".")[0]
        f_name = os.path.join(xmlpath, f_name)
        f_name = f_name.replace(os.sep, '/')
        ground_truth = groundtruthutils.GroundTruthXml(f_name)

        return ground_truth, file_name_for_metrics
