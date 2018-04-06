from xml.dom import minidom
import cv2
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjectionTree
from redis import Redis
import os
from nearpy.storage import RedisStorage
filename = 'MVI_20011.xml'
XMLPATH='/media/student/DATA3/data_pool2/NVIDIA_AICity/XML'
IMGPATH='/media/student/DATA3/data_pool2/NVIDIA_AICity/Insight-MVT_Annotation_Train'
SPATH = 'CROPPED_2012'
DIMENSION = 3072
redis_object = Redis(host='localhost', port=6379, db=1)
redis_storage = RedisStorage(redis_object)


VIDEOS = ['MVI_20012', 'MVI_20032', 'MVI_20033', 'MVI_20034', 'MVI_20035', 'MVI_20051', 'MVI_20052', 'MVI_20061', 'MVI_20062', 'MVI_20063', 'MVI_20064', 'MVI_20065']
for filename in VIDEOS:
    storage_name = filename.split('.')[0]

    config = redis_storage.load_hash_configuration(storage_name)
    lshash = None
    if config is None:
        lshash = RandomBinaryProjectionTree(storage_name, 5, 30)
    else:
        lshash = RandomBinaryProjectionTree(None, None)
        lshash.apply_config(config)

    engine = Engine(DIMENSION, lshashes=[lshash], storage=redis_storage)

    doc = minidom.parse(os.path.join(XMLPATH,filename + '.xml'))
    FILE = filename.split('.')[0]
    image_s_path = IMGPATH + '/' + FILE + '/img'
    itemlist = doc.getElementsByTagName('frame')
    for frame in itemlist:
        frameID = str(frame.attributes['num'].value)
        image_path = image_s_path + frameID.zfill(5) +'.jpg'
        image = cv2.imread(image_path)
        targetlist = frame.getElementsByTagName('target')
        for target in targetlist:
            targetID = target.attributes['id'].value
            box = target.getElementsByTagName('box')
            X = int(float(box[0].attributes['left'].value))
            Y = int(float(box[0].attributes['top'].value))
            W = int(float(box[0].attributes['width'].value))
            H = int(float(box[0].attributes['height'].value))
            attribute = target.getElementsByTagName('attribute')
            speed = float(attribute[0].attributes['speed'].value)

            if (W >= 100 and speed > 1.0):
                save_name = FILE + '_' + str(frameID) +  '_' + targetID + '_' + str(X) + '_' + str(Y) + '_' + str(W) + '_' + str(H) + '_' + str(speed)
                print(save_name)
                ROI = image[Y:Y+W,X:X+W]
                reducedROI = cv2.resize(ROI, (32,32))
                flatROI = reducedROI.flatten()
                engine.store_vector(flatROI, save_name)

    redis_storage.store_hash_configuration(lshash)
    print(filename)
