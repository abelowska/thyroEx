import os
import xml.etree.cElementTree as ET
import re

path = ''

files_list = os.listdir(path)  # returns list


for file in files_list:
    filename, file_extension = os.path.splitext(file)
    if file_extension == '.xml':
        tree = ET.parse(path + file)
        tirads = tree.findall('tirads')[0].text
        print(tirads)
        if tirads is None:
            pass
        else:
            for f in files_list:
                matcher = re.compile(r'\b{}_'.format(filename))
                if matcher.match(f):
                    print('     {}'.format(f))
                    new_path = '{}{}/{}'.format(path, tirads, f)
                    os.rename(path + f, new_path)




