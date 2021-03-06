import os
class DataWriter:
    def __init__(self, dir_path, prefix, max_num) -> None:
        self.dir_path = dir_path
        self.prefix = prefix
        self.max_num = max_num
        self.files = []
        dir = os.path.join(dir_path, prefix)
        file_prefix = os.path.join(dir, prefix)
        if not os.path.exists(dir):
            os.mkdir(dir)
        for i in range(max_num):
            file_name = '{}_{}.log'.format(file_prefix, i)
            self.files.append(open(file_name, 'x'))
    
    def write(self, idx, str):
        self.files[idx].write('{}\n'.format(str))        
