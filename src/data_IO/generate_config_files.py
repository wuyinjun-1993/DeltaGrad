'''
Created on Mar 15, 2019

'''

import json
import sys


import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_IO.Load_data import git_ignore_folder
except ImportError:
    from Load_data import git_ignore_folder


def write_to_file(map, config_file_name):
    with open(config_file_name, 'w') as outfile:
        json.dump(map, outfile, indent=4)



directory_name = '../../../data/'

config_file_name = '../train_data_meta_info.ini'


sys_args = sys.argv

if len(sys_args) > 1:
    git_ignore_folder = sys_args[1]
    if len(sys_args) > 2:
        directory_name = sys_args[2]



file_name = directory_name + 'szeged-weather/weatherHistory.csv'

x_cols = [5,6,7,8]
 
y_cols = [3,4]

from_csv = True

map = {}

map[file_name] = {'x_cols': x_cols, 'y_cols': y_cols, 'from_csv': from_csv}




file_name = directory_name + 'candy/candy-data.csv'

x_cols = [2,3,4,5,6,7,8,9,10,11,12]
  
y_cols = [1]

from_csv = True

map[file_name] = {'x_cols': x_cols, 'y_cols': y_cols, 'from_csv': from_csv}



file_name = directory_name + 'BlogFeedback/blogData_train.csv'

x_cols = list(range(280))
   
# x_cols = [x+1 for x in x_cols]
  
y_cols = [280]

from_csv = True

map[file_name] = {'x_cols': x_cols, 'y_cols': y_cols, 'from_csv': from_csv}






file_name = directory_name + 'adult.csv'

x_cols = [0, 4, 10, 11, 12]
      
y_cols = [14]

from_csv = True

map[file_name] = {'x_cols': x_cols, 'y_cols': y_cols, 'from_csv': from_csv}


file_name = directory_name + 'toxic/train.csv'

x_cols = [3, 4, 5, 6, 7]
       
y_cols = [2]

from_csv = True

map[file_name] = {'x_cols': x_cols, 'y_cols': y_cols, 'from_csv': from_csv}

file_name = directory_name + 'toxic/test_labels.csv'

x_cols = [2, 3, 4, 5, 6]
       
y_cols = [1]

from_csv = True

map[file_name] = {'x_cols': x_cols, 'y_cols': y_cols, 'from_csv': from_csv}











file_name = directory_name + 'credit_card/creditcard.csv'

x_cols = list(range(29))
   
x_cols = [x+1 for x in x_cols]
   
y_cols = [30]

from_csv = True

map[file_name] = {'x_cols': x_cols, 'y_cols': y_cols, 'from_csv': from_csv}





file_name = directory_name + 'cifar10'

from_csv = False

map[file_name] = {'feature_num': 3072, 'from_csv': from_csv}









file_name = directory_name + 'HIGGS'

from_csv = False

map[file_name] = {'feature_num': 28, 'from_csv': from_csv}



file_name = directory_name + 'aloi.scale'

from_csv = False

map[file_name] = {'feature_num': 128, 'from_csv': from_csv}




file_name = directory_name + 'minist.csv'
 
y_cols = [0]
 
x_cols = list(range(784))
 
x_cols = [x+1 for x in x_cols]

from_csv = True

map[file_name] = {'x_cols': x_cols, 'y_cols': y_cols, 'from_csv': from_csv}

file_name = directory_name + 'heartbeat/mitbih_train.csv'
 
y_cols = [187]
 
x_cols = list(range(187))
 
# x_cols = [x+1 for x in x_cols]
from_csv = True

map[file_name] = {'x_cols': x_cols, 'y_cols': y_cols, 'from_csv': from_csv}




file_name = directory_name + 'sgemm_product.csv'
 
y_cols = [14, 15, 16, 17]
 
x_cols = list(range(14))
 
# x_cols = [x+1 for x in x_cols]
from_csv = True

map[file_name] = {'x_cols': x_cols, 'y_cols': y_cols, 'from_csv': from_csv}










file_name = directory_name + 'Sensorless.scale'
 
# x_cols = [x+1 for x in x_cols]

from_csv = False

map[file_name] = {'feature_num': 48, 'from_csv': from_csv}

file_name = directory_name + 'shuttle.scale.tr'
 
# x_cols = [x+1 for x in x_cols]


from_csv = False

map[file_name] = {'feature_num': 9, 'from_csv': from_csv}



file_name = directory_name + 'covtype'
 
# x_cols = [x+1 for x in x_cols]

from_csv = False

map[file_name] = {'feature_num': 54, 'from_csv': from_csv}


file_name = directory_name + 'skin_nonskin'
 
# x_cols = [x+1 for x in x_cols]

from_csv = False

map[file_name] = {'feature_num': 3, 'from_csv': from_csv}






file_name = directory_name + 'covtype_binary'
 
# x_cols = [x+1 for x in x_cols]

from_csv = False

map[file_name] = {'feature_num': 54, 'from_csv': from_csv}



file_name = directory_name + 'rcv1_train.multiclass'
 
# x_cols = [x+1 for x in x_cols]

from_csv = False

map[file_name] = {'feature_num': 47236, 'from_csv': from_csv}



file_name = directory_name + 'rcv1_test.multiclass'
 
# x_cols = [x+1 for x in x_cols]

from_csv = False

map[file_name] = {'feature_num': 47236, 'from_csv': from_csv}






map['git_ignore_folder'] = git_ignore_folder

map['directory'] = directory_name

write_to_file(map, config_file_name)


