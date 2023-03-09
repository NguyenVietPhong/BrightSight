import os


class Path():
    def __init__(self, file):
        self.path_current = os.path.dirname(os.path.abspath(__file__))
        self.file = file

    def cat_path(self):
        return os.path.join(self.path_current, self.file)
    



class Logger():
    def __init__(self, path=None):
        self.path = path
    
    def write_logger(self, line):
        with open(self.path, 'a') as f:
            f.writelines(line)

    