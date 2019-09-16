import sys
sys.path.append('/export/home/suny/VideoSMPL/src')
from dataset.image_base import *

class Video_base(Image_base):
    def __init__(self, train_flag=True,video=False,get_features=False,feature_name='bilinear',receptive_field = 81,spawn = 9,high_resolution=False):
        super(Video_base,self).__init__(train_flag,high_resolution)
        self.video=video
        self.receptive_field = receptive_field
        self.spawn = spawn
        self.get_features = get_features
        if args.save_features:
            self.train_flag=False
        if self.video:
            self.video_frame_spawn = receptive_field
            self.train_flag=False

    def get_item_video(self,index):
        raise NotImplemented

    def __len__(self):
        if self.video:
            return self.video_count
        return len(self.file_paths)
    
    def __getitem__(self, index):
        if self.video:
            return self.get_item_video(index)
        else:
            return self.get_item_single_frame(index)

