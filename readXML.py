from xml.dom import minidom
import cv2

filename = 'MVI_20011.xml'

doc = minidom.parse(filename)
image_s_path = 'MVI_20011/img'
itemlist = doc.getElementsByTagName('frame')
print(len(itemlist))
for item in itemlist:
    targetlist = item.getElementsByTagName('target')
    #target_id = target.getElementsByTagName('target')
    i = 1
    for target in targetlist:
        targetID = str(i).zfill(5)
        image_path = image_s_path + targetID +'.jpg'
        print(image_path)
        image = cv2.imread(image_path)
        cv2.imshow('image', image)
        box = target.getElementsByTagName('box')
        X = int(float(box[0].attributes['left'].value))
        Y = int(float(box[0].attributes['top'].value))
        W = int(float(box[0].attributes['width'].value))
        H = int(float(box[0].attributes['height'].value))
        attribute = target.getElementsByTagName('attribute')
        speed = int(float(attribute[0].attributes['speed'].value))
        if (W >= 100 and speed > 1):
            ROI = image[Y:Y+W,X:X+W]
            cv2.imshow('ROI', ROI)
            cv2.waitKey()
        i = i + 1
    cv2.destroyAllWindows()
    
