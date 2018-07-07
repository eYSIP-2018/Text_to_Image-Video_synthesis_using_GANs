import os
import cv2
import numpy as np
import h5py
from PIL import Image,ImageSequence
from torchvision.datasets import ImageFolder
import tqdm
import zipfile
from requests import get

"""
Description:
This file will download the dataset and unzip it and then preprocess it as required.
once you run this file dataset would be prepared in the kth folder , and it will need 
to be copied to data/actions 
"""




os.makedirs('kth',exist_ok=True)

def download(url, file_name):
    with open(file_name, "wb") as file:
        print("downloading dataset file",file_name)
        response = get(url)
        file.write(response.content)
    print("{} downloading complete".format(file_name))
    with zipfile.ZipFile(file_name, "r") as zip_ref:
        zip_ref.extractall('kth')

def convertFile(inputpath, targetFormat='.gif'):
    outputpath = os.path.splitext(inputpath)[0] + targetFormat
    reader = imageio.get_reader(inputpath)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(outputpath, fps=fps)
    for i,im in enumerate(reader):
        sys.stdout.write("\rframe {0}".format(i))
        sys.stdout.flush()
        writer.append_data(im)
    writer.close()

dict  = {'boxing':0,'handclapping':1,'handwaving':2,'jogging':3,'running':4,'walking':5}
for i in dict.keys():
    download('http://www.nada.kth.se/cvap/actions/{}.zip'.format(i),'kth_{}.zip'.format(i))

for i in os.listdir('kth'):
    convertFile(i)

for idx, entry in enumerate(os.listdir('kth')):
    if 'd2' in entry:
        continue
    parts = entry.split("_")
    os.makedirs("kth_out/{0}".format(lower(dict[parts[1]])), exist_ok=True)
    try:
        imageObject = Image.open("kth/{0}".format(entry))
        vid_iter = ImageSequence.Iterator(imageObject)
        vid_frames = [cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_BGR2RGB) for img in vid_iter]
        vid_frames = [cv2.copyMakeBorder(img, 20, 20, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]) for img in vid_frames]
        idy = str(idx + 100000001)
        idy = idy[1:]
        image = np.hstack(vid_frames)
        cv2.imwrite("kth_out/{0}/{1}.png".format(parts[1],idy),image)
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        print("error is", e)
        pass






