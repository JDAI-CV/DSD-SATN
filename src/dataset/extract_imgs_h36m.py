import cv2
import sys,os
import glob
from shutil import copyfile


cam_dict = {'54138969':0, '55011271':1, '58860488':2, '60457274':3}
subject_id = 'S9'

def extract_imgs(subject_id):
    video_files = glob.glob('archives/{}/Videos/*.mp4'.format(subject_id))
    for video_file in video_files:
        video_file = video_file.replace(' ', '\ ')
        video_name = os.path.basename(video_file)
        
        action_name, cam_str, _ = video_name.split('.')
        cam_id = cam_dict[cam_str]
        target_name = 'archives/images/{}_{}_{}'.format(subject_id, action_name, cam_id)

        print(video_file, target_name)
        cmd = 'ffmpeg -i {} -f image2 -r 10 -qscale:v 2 {}'.format(video_file, target_name) + '_%d.jpg'
        print(cmd)
        os.system(cmd)

def fix_eval():
    images = glob.glob('archives/image/*.jpg')
    for image in images:
        basename =  os.path.basename(image)
        subject_id, action_name, cam_id, frame_id = basename.split('_')
        # I have dropped the first 20 frames for remove redundency
        frame_id = int(frame_id.replace('.jpg', ''))+3
        correct_img_path = os.path.join('archives', 'images', '{}_{}_{}_{}.jpg'.format(subject_id, action_name, cam_id, frame_id))
        target_path = os.path.join('archives', 'corrected_imgs', basename)
        if os.path.exists(correct_img_path):
            copyfile(correct_img_path, target_path)
        else:
            print('missing', basename)

if __name__ == '__main__':
    extract_imgs('S9')