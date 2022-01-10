import configparser
import os


class ConfigurationParser:
    def __init__(self, filename, section):
        self.filename = filename
        self.section = section
        self.section_config_dict = self.get_dict(section)
        self.general_config_dict = self.get_dict('GENERAL')

    def get_dict(self, section):
        config = configparser.RawConfigParser()
        config.read(self.filename)
        return dict(config.items(section))

    @property
    def data_path(self):
        return os.path.join(self.home_path, self.section_config_dict['data_path'])

    @property
    def annotations_path(self):
        return os.path.join(self.home_path, self.section_config_dict['annotations_path'])

    @property
    def home_path(self):
        return self.general_config_dict['home_path']

    @property
    def model_path(self):
        return self.general_config_dict['model_path']

    @property
    def coral_threshold(self):
        return float(self.general_config_dict['coral_threshold'])



if __name__ == '__main__':

    parser = ConfigurationParser("../config/config.txt", 'SHAD')

