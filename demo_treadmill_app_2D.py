import cv2
import imutils
import numpy as np
from glob import glob
from Custom import AdjustPredictionLayer
from ford_utils import read_video_configs, put_text

from efficientnet.model import *
from efficientnet.preprocessing import center_crop_and_resize
from efficientnet.preprocessing import preprocess_input
from keras.models import load_model

import os
from time import time

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


import argparse
import torch



parser = argparse.ArgumentParser(description='Testing efficient networks')
parser.add_argument('--workers', default=1, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--dataset', default='imagenet', help='Name of the dataset')
parser.add_argument('--batch-size', default=1, type=int, help='mini-batch size (default: 512)')
parser.add_argument('--num-classes', default=1, type=int, help='# of classes in the dataset')
parser.add_argument('--s', default=0.2, type=float, help='Width scaling factor')

# weights_path = '/home/tan/William/DiCENeT/EdgeNets/results_classification_main/dgx/model_dicenet_Heart/aug_0.2_1.0/s_0.2_inp_96_sch_hybrid/20190826-152157/dicenet_0.2_best.pth'
# weights_path = '/home/tan/William/DiCENeT/EdgeNets/results_classification_main/k9_model_dicenet_Heart/aug_0.2_1.0/s_0.2_inp_96_sch_hybrid/20190911-142706/dicenet_0.2_best.pth'

weights_path_hc = '/home/tan/William/DiCENeT/EdgeNets/results_classification_main/model_hc2_dicenet_Heart/aug_0.2_1.0/s_0.2_inp_96_sch_hybrid/20190908-115854/dicenet_0.2_best.pth'
weights_path_wc = '/home/tan/William/DiCENeT/EdgeNets/results_classification_main/model_wc2_dicenet_Heart/aug_0.2_1.0/s_0.2_inp_96_sch_hybrid/20190908-120109/dicenet_0.2_best.pth'

# weights_path = '/home/tan/William/DiCENeT/CardioNet/dicenet_0906_0.2_best.pth'
parser.add_argument('--weights', type=str, default='', help='weight file')
parser.add_argument('--inpSize', default=96, type=int, help='Input size')
parser.add_argument('--model', default='dicenet', help='Which model?')
parser.add_argument('--model-width', default=96, type=int, help='Model width')
parser.add_argument('--model-height', default=96, type=int, help='Model height')
parser.add_argument('--channels', default=70, type=int, help='Input channels')
parser.add_argument('--people', default='1', type=int, help='people')
parser.add_argument('--mdtime',default='time',help='model-date')
parser.add_argument('--wtpath',type=str,default='path',help='weight-path')
parser.add_argument('--ksize', default=9, type=int,help='Kernel_size for convolution')
args = parser.parse_args()

#k3_1e4_s0.2
weights_path = '/home/tan/William/DiCENeT/EdgeNets/results_classification_main/model_many0920/results_classification_main/k3_1e4_model_dicenet_Heart/aug_0.2_1.0/s_0.2_inp_96_sch_hybrid/20190917-012140/dicenet_0.2_best.pth'
weights_path = args.wtpath
args.weights = args.wtpath

if args.model == 'dicenet':
    from model.classification import dicenet as net
    args.weights = weights_path
elif args.model == 'dicenet_hc':
    from model.classification import dicenet_hc as net
    args.weights = weights_path_hc
elif args.model =='dicenet_wc':
    from model.classification import dicenet_wc as net
    args.weights = weights_path_wc

print(args.weights)





class HeartRateEstimator(object):
    def __init__(self, **kwargs):
        # load models
        self.hr_model_path = kwargs.get('hr_model_path')
        self.face_detection_model_path = kwargs.get('face_detection_model_path')
        # self.hr_model = load_model(self.hr_model_path)
        print('//////load face model start//////')
        self.face_detection_model = load_model(self.face_detection_model_path, custom_objects={'AdjustPredictionLayer': AdjustPredictionLayer})
        print('//////load face model done//////')
        # optional_settings
        self.desired_signal_len = 70
        self.use_imutils_rotate = False
        self.rotate_angle = -90
        self.face_model_input_size = (225, 225)  # (w, h) 480, 270
        self.face_model_threshold = 0.5
        self.hr_model_input_size = (96, 96)
        self.update_period = 7  # frames
        self.smooth_range = 5
        self.data_std = 36.913
        self.data_mean = 117.662

        # variables initialization
        self.hrs = []
        self.g_values = []
        self.frame_cnt = 0
        self.hr_to_show = ''

        self.lastface = None

        # heartrate model setting  
        self.hr_model = net.CNNModel(args)
        self.num_gpus = torch.cuda.device_count()
        self.device = 'cuda' if self.num_gpus >=1 else 'cpu'
        self.weight_dict = torch.load(args.weights, map_location=torch.device(self.device))
        self.hr_model.load_state_dict(self.weight_dict)

        if self.num_gpus >= 1:
            self.hr_model = torch.nn.DataParallel(self.hr_model)
            self.hr_model = self.hr_model.cuda()
            if torch.backends.cudnn.is_available():
                import torch.backends.cudnn as cudnn
                cudnn.benchmark = True
                cudnn.deterministic = True
        print('////HRE init done////')


    def estimate(self,face_video, imageBGR):
        self.frame_cnt += 1
        video_height, video_width, _ = imageBGR.shape
        if self.use_imutils_rotate:
            imageBGR = imutils.rotate(imageBGR, self.rotate_angle)
            # video_height, video_width = video_width, video_height

        canvas = imageBGR.copy()
        detected_face = self.face_detection(imageBGR)

        if len(detected_face) == 0:
            print('[*] Face not found, skipped 1 frame.')
            self.g_values.append([])
        else:
            # #sort faces by confidence and select the face with the highest confidence
            # detected_face = sorted(detected_face, key=lambda x: x[0], reverse=True)[0]
            if self.frame_cnt <= 1:
                detected_face = detected_face[0]
                self.lastface = detected_face
            else:
                if detected_face[0][1] >= (self.lastface[1] + self.lastface[3]*2) or detected_face[0][2] >= (self.lastface[2] + self.lastface[4]*2):
                    try:
                        detected_face = detected_face[1]
                    except:
                        detected_face = self.lastface
                    self.lastface = detected_face
                else:
                    detected_face = detected_face[0]
                    self.lastface = detected_face
            (roi_x1, roi_y1), (roi_x2, roi_y2), (face_x1, face_y1), (face_x2, face_y2) = self.translate2Coordinate(detected_face, video_width, video_height)
            canvas = cv2.rectangle(canvas, (face_x1, face_y1), (face_x2, face_y2), (0, 255, 255), 3)
            canvas = cv2.rectangle(canvas, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 3)

            roi = imageBGR[roi_y1:roi_y2, roi_x1:roi_x2, :] / 255.
            h_ , w_ , c_ = roi.shape
            if h_ ==0 or w_ == 0:
                self.g_values.append([])
            # print(roi.shape)
            else:
                _, g_channel, _ = cv2.split(roi)
                # g_value = np.mean(g_channel)
                resize_green = cv2.resize(g_channel, self.hr_model_input_size)
                self.g_values.append(resize_green)

        if self.frame_cnt % self.update_period == 0:
            hr_prediction = self.estimate_hr()
            if hr_prediction:
                self.hrs.append(hr_prediction)
                self.hr_to_show = int(np.mean(self.hrs[-self.smooth_range:]))

        try:
            face_video.write(canvas)
        except:
            face_video.write(imageBGR)
        canvas = put_text(canvas, '{}'.format(self.hr_to_show), (10, 100), text_color=(255, 255, 255), font_scale=2, text_thickness=3)

        cv2.namedWindow('canvas', cv2.WINDOW_NORMAL)
        cv2.imshow('canvas', canvas)
        cv2.waitKey(1)

        return self.hr_to_show, canvas

    def estimate_hr(self):
        signal = self.g_values[-self.desired_signal_len:]

        if len(signal) < self.desired_signal_len or [] in signal:
            return []

        hr_model_input = np.expand_dims(signal, axis = 0)
        hr_model_input = torch.cuda.FloatTensor(hr_model_input)
        # hr_model_input = np.reshape(hr_model_input, (-1, self.hr_model_input_size[0] * self.hr_model_input_size[1], self.desired_signal_len))
        hr_prediction = self.hr_model(hr_model_input)
        hr_prediction = (hr_prediction * self.data_std) + self.data_mean
        hr_prediction = round(float(hr_prediction))

        return hr_prediction

    def face_detection(self, imageBGR):
        model_input = cv2.resize(imageBGR, self.face_model_input_size)

        # BGR -> RGB
        model_input = model_input[:, :, ::-1]
        model_input = np.expand_dims(model_input, 0) / 255.
        prediction = self.face_detection_model.predict(model_input)[0]
        prediction = list(filter(lambda x: x[0] > self.face_model_threshold, prediction))

        return prediction

    def translate2Coordinate(self, detected_face, width, height):
        head_center_x, head_center_y, head_w, head_h = detected_face[1:]

        # head_w *= 1.5
        # head_h *= 0.5

        face_x1 = int((head_center_x - head_w / 2) * float(width))
        face_y1 = int((head_center_y - head_h / 2) * float(height))
        face_x2 = int((head_center_x + head_w / 2) * float(width))
        face_y2 = int((head_center_y + head_h / 2) * float(height))

        roi_w = head_w #* 0.7
        roi_h = head_h #* 0.4

        roi_x1 = int((head_center_x - roi_w / 2) * float(width))
        roi_y1 = int((head_center_y - roi_h / 2) * float(height))
        roi_x2 = int((head_center_x + roi_w / 2) * float(width))
        roi_y2 = int((head_center_y + roi_h / 2) * float(height))

        return (roi_x1, roi_y1), (roi_x2, roi_y2), (face_x1, face_y1), (face_x2, face_y2)



# video_path = glob('/media/ford/320g/ford_data/rgb_hr/data/data0827/sub1/100_1/*.mp4')[0]
# video_reader = cv2.VideoCapture(video_path)
# video_reader = cv2.VideoCapture(0)
# video_configs = read_video_configs(video_reader)
# video_frame_num = video_configs['video_frame_num']

if args.model == 'dicenet':
    hr_model_path = 'models_HeartRate/DiCENeT'
elif args.model == 'dicenet_hc':
    hr_model_path = 'models_HeartRate/DiCENeT_hc'
elif args.model =='dicenet_wc':
    hr_model_path = 'models_HeartRate/DiCENeT_wc'


face_detection_model_path = 'models_Face/Face_EfficientNetB0/Face_widerface200.h5'
heartRateEstimator = HeartRateEstimator(face_detection_model_path=face_detection_model_path)

# # for usb cam
# while True:
#     success, imageBGR = video_reader.read()
#     if not success:
#         break
#
#     hr = heartRateEstimator.estimate(imageBGR=imageBGR)

people = ['sub00', 'sub01', 'sub02','sub03','sub04','sub05', 'sub06','sub07','sub99']

people = people[args.people]
video_root = os.path.join('../HeartRate_app/video_records', people)
#video_root = os.path.join('../TX2_0925/video_records', people)
signal_length = '70'
model_date = args.mdtime
face_model_folder = face_detection_model_path.split('/')[1]
hr_model_folder = hr_model_path.split('/')[1]
outdir = os.path.join('accuracy/prediction/{}/{}'.format(face_model_folder, hr_model_folder + model_date ),
                      signal_length, people)
if not os.path.isdir(outdir):
    os.makedirs(outdir)
outdir_video = os.path.join('result/face_video/{}/{}'.format(face_model_folder, people))
if not os.path.isdir(outdir_video):
    os.makedirs(outdir_video)
outdir_fps = os.path.join('accuracy/HeartRate_FPS/{}/{}/{}/{}'.format(face_model_folder, hr_model_folder + model_date, signal_length, people))
if not os.path.isdir(outdir_fps):
    os.makedirs(outdir_fps)


for video_path in glob(os.path.join(video_root, '*mp4')):
    video_name = video_path.split('/')[-1]
    txt_name = video_name.replace('ori', 'pred').replace('mp4', 'txt')

    vid = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    face_video = cv2.VideoWriter(os.path.join(outdir_video, video_name), fourcc, 7, video_size)
    txt_fps = open(os.path.join(outdir_fps, txt_name), 'w')
    FPS = []
    with open(os.path.join(outdir , '{}'.format(txt_name)), 'w') as f:

        while True:
            ret, frame = vid.read()
            if not ret:
                break
            # frame = np.rot270(frame)
            tic1 = time()
            
            hr, _ = heartRateEstimator.estimate(face_video, imageBGR=frame)
            toc1 = time() - tic1
            fps = 1 / toc1
            print(fps)
            FPS.append(fps)
            txt_fps.write('{:.2f}'.format(fps))
            txt_fps.write('\n')


            f.write('{}\n'.format(hr))

    txt_fps.write('Average FPS: {:.2f}'.format(np.mean(np.asarray(FPS))))
    txt_fps.close()
    face_video.release()
