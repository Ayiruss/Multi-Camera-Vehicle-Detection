from xml.dom import minidom
import cv2
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from redis import Redis
import os
from nearpy.storage import RedisStorage
filename = 'MVI_20011.xml'
XMLPATH='/home/student/data_pool2/NVIDIA_AICity_Dataset/DETRAC-Train-Annotations-XML'
IMGPATH='/media/student/DATA3/data_pool2/NVIDIA_AICity/'
SPATH = 'SEARCH_DIR'
DIMENSION = 3072
redis_object = Redis(host='localhost', port=6379, db=0)
redis_storage = RedisStorage(redis_object)

config = redis_storage.load_hash_configuration('MyHash')
lshash = None
if config is None:
    lshash = RandomBinaryProjections('MyHash', 10)
else:
    lshash = RandomBinaryProjections(None, None)
    lshash.apply_config(config)

engine = Engine(DIMENSION, lshashes=[lshash], storage=redis_storage)


#for filename in os.listdir(XMLPATH):
doc = minidom.parse(os.path.join(XMLPATH,filename))
FILE = filename.split('.')[0]
image_s_path = IMGPATH + '/' + FILE
itemlist = doc.getElementsByTagName('frame')
#print(len(itemlist))
frame_id = 1
for item in itemlist:
    print(item.attributes['num'].value)

#redis_storage.store_hash_configuration(lshash)
