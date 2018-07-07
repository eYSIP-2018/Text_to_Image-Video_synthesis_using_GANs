"""

Usage:
    generate_videos.py [options] <model> <output_folder>

Options:
    -n, --num_videos=<count>                number of videos to generate [default: 10]
    -o, --output_format=<ext>               save videos as [default: gif]
    -f, --number_of_frames=<count>          generate videos with that many frames [default: 16]
    -c, --category=<category>               generate videos for a specific category [default: running]
    
"""

import os
import docopt
import torch
import numpy as np
from trainers import videos_to_numpy
import cv2
import subprocess as sp
import imageio


if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    generator = torch.load(args["<model>"])
    generator.eval()
    num_videos = int(args['--num_videos'])
    output_folder = args['<output_folder>']
    category = args['--category'].lower()
        
    dict = {'boxing':0,'handclapping':1,'handwaving':2,'jogging':3,'running':4,'walking':5}
    c = dict[category]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)        
        
    for i in range(num_videos):
        v, cats = generator.sample_videos(1, int(args['--number_of_frames']),category=c)
        cat = cats.data.cpu().numpy()
        video = videos_to_numpy(v).squeeze().transpose((1, 2, 3, 0))
        imageio.mimsave("video{0}_{1}.gif".format(i,category),video)
#        save_video(args["--ffmpeg"], video, os.path.join(output_folder, "{}.{}".format(i, args['--output_format'])))
