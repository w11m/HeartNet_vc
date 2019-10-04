import cv2
import os
from glob import glob
import numpy as np
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
# plt.ion()
import csv

parser = argparse.ArgumentParser(description='Testing efficient networks')
parser.add_argument('--people', default='8', type=int, help='people')
parser.add_argument('--mdtime', default='a0924_k9_1e03_s075', help='mdtimee')
args = parser.parse_args()

people = people = ['sub00', 'sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07','sub99']
people = people[args.people]

signal_length = '70'
cats = ['ground_truth', 'prediction']
model_date = args.mdtime
hr_mode = 'DiCENeT'
face_mode = 'Face_EfficientNetB0'

# indir_video = os.path.join('../HeartRate_TX2/video_record', people)
outdir_video = os.path.join('result/video_prediction', face_mode, hr_mode + model_date, signal_length, people)
if not os.path.isdir(outdir_video):
    os.makedirs(outdir_video)

indir_pred = os.path.join('accuracy', cats[1], face_mode, hr_mode + model_date, signal_length, people)
outdir_error = os.path.join('result/error_graph', face_mode, hr_mode + model_date, signal_length, people)
if not os.path.isdir(outdir_error):
    os.makedirs(outdir_error)

outdir_table = os.path.join('result/table', face_mode, hr_mode + model_date, signal_length, people)
if not os.path.isdir(outdir_table):
    os.makedirs(outdir_table)

fps_cam = 30  # no use
fps_pred = 7
rm_first_sec = 15  # remove first num second
rm_last_sec = 15
pre_empty_frame = 35  # previous num frame are empty
date = 'treadmill'
stat = 'Dice' + args.mdtime

for predpath in glob(os.path.join(indir_pred, '*txt')):
    filename = predpath.split('/')[-1].replace('_pred', '')
    dist, videoname = filename[:-4].split('_')
    txt_filename = '20190910_{}_150_gt'.format(videoname) + '.txt'
    gtpath = os.path.join('../HeartRate_app/video_gts', people, txt_filename)

    invideo = cv2.VideoCapture(os.path.join('result/face_video', face_mode, people, filename[:-4] + '_ori.mp4'))
    width, height = int(invideo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(invideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outvideo = cv2.VideoWriter(os.path.join(outdir_video, filename[:-4]) + '.mp4', fourcc, fps_pred, (width, height))

    with open(gtpath, 'r') as ground_truth:
        ground_truth = ground_truth.readlines()
    with open(predpath, 'r') as prediction:
        prediction = prediction.readlines()
    if people == 'sub00' or people == 'data0828_sub5' or people == 'data0829_sub3':
        if videoname == 'dynamic':
            del prediction[:int(signal_length)]
    elif people == 'data0828_sub6' or people == 'data0828_sub1':
        if videoname == '75':
            del prediction[:int(signal_length)]

    # get heart-rate
    gt = []
    for i in range(1, len(ground_truth), 3):
        heat_rate = ground_truth[i].split('\n')[0]
        gt.append(int(heat_rate))
    pred = []
    for num, heat_rate in enumerate(prediction):
        try:
            heat_rate = heat_rate.split('\n')[0]
            print((heat_rate), num)
            pred.append(int(heat_rate))
        except:
            pass

    # remove first num second and last num second
    noise1 = rm_first_sec
    del gt[:noise1]
    noise1 = rm_first_sec * fps_pred
    del pred[:noise1]
    noise2 = rm_last_sec
    del gt[-noise2:]
    noise2 = rm_last_sec * fps_pred
    del pred[-noise2:]

    # create new GT correspond to prediction
    gt_new = []
    for heat_rate in gt:
        for i in range(fps_pred):
            gt_new.append(heat_rate)

    # remove last prediction weren't correspond to GT
    no_need = len(gt_new) - len(pred)
    if no_need > 0:
        del gt_new[-no_need:]
    elif no_need < 0:
        del pred[no_need:]

    # draw error grapg
    # error = list(map(lambda distance: distance[0]-distance[1], zip(pred, gt_new)))    #much slower
    avg_pred, avg_gt = np.mean(np.asarray(pred)), np.mean(np.asarray(gt_new))
    error = [pred[i] - gt_new[i] for i in range(len(gt_new))]  # faster
    error_avg = np.mean(abs(np.asarray(pred) - np.asarray(gt_new)))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(gt_new)
    plt.plot(pred)
    plt.legend(['ground truth', 'prediction'], loc='upper left')
    plt.title('HeartRate: model={}, status={}'.format(stat, str(videoname)))
    plt.xlabel('time')
    plt.ylabel('bpm')
    plt.text(len(gt_new), max(max(pred), max(gt_new)),
             'averge ground truth: {:.4}'.format(avg_gt), ha='right', va='top',
             bbox=dict(boxstyle='round', ec=(1, 0, 0), fc=(1, 1, 1)))
    plt.text(len(pred), max(max(pred), max(gt_new)) - 7,
             'averge prediction: {:.4}'.format(avg_pred), ha='right', va='top',
             bbox=dict(boxstyle='round', ec=(1, 0, 0), fc=(1, 1, 1)))

    plt.subplot(2, 1, 2)
    plt.plot(error, color='blueviolet')  # purple line
    plt.legend(['error'], loc='upper left')
    plt.title('HeartRate Error: model={}, status={}'.format(stat, str(videoname)))
    plt.xlabel('time')
    plt.ylabel('bpm')
    plt.text(len(error), max(error),
             'averge error: {:.4}'.format(error_avg), ha='right', va='top',
             bbox=dict(boxstyle='round', ec=(1, 0, 0), fc=(1, 1, 1)))

    plt.draw()
    plt.subplots_adjust(hspace=0.3)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.pause(0.001)
    plt.savefig(os.path.join(outdir_error, '{}_{}_{}.jpg'.format(date, stat, videoname)), bbox_inches='tight')
    cv2.destroyAllWindows()

    # Create prediction video
    cv2.namedWindow('Result', 0)

    with open(os.path.join('accuracy/HeartRate_FPS', face_mode, hr_mode + model_date, signal_length, people,
                           filename[:-4] + '.txt'), 'r') as txt_time:
        txt_time = txt_time.readlines()
    del txt_time[-1]
    times = [float(time.split('\n')[0]) for time in txt_time]
    fps = times[:len(pred)]
    for i in range(len(pred)):
        ret, frame = invideo.read()
        if not ret:
            break

        if i < pre_empty_frame:
            outvideo.write(frame)
            continue

        txtx_size, _ = cv2.getTextSize('Ground Truth: {:.2f}'.format(gt_new[i]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                       fontScale=0.5, thickness=1)
        cv2.rectangle(frame, (0, 0), (txtx_size[0] + 5 * 2, txtx_size[1] * 4 + 3 * 5), color=(0, 0, 0), thickness=-1)
        cv2.putText(frame, 'Ground Truth: {:.2f}'.format(gt_new[i]), (5, (txtx_size[1] + int(1.5 * 2)) * 1),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 255), thickness=1)
        cv2.putText(frame, 'Prediction: {:.2f}'.format(pred[i]), (5, (txtx_size[1] + int(1.5 * 2)) * 2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 255), thickness=1)
        cv2.putText(frame, 'Error: {:.2f}'.format(error[i]), (5, (txtx_size[1] + int(1.5 * 2)) * 3),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 255), thickness=1)
        cv2.putText(frame, 'FPS: {:.2f}'.format(fps[i]), (5, (txtx_size[1] + int(1.5 * 2)) * 4),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 255), thickness=1)

        cv2.imshow('Result', frame)
        cv2.waitKey(1)
        outvideo.write(frame)
    invideo.release()
    outvideo.release()
    cv2.destroyAllWindows()

    # Create csv files
    gt_new = np.array(gt_new)
    pred = np.array(pred)
    precent_deviation = 100 * abs(pred - gt_new) / gt_new
    fps = np.array(fps)
    with open(os.path.join(outdir_table, '{}.csv'.format(filename)), 'w', newline='') as f:
        head = ['Frame Number', 'Ground Truth', 'Prediction', 'Error', 'Percent Deviation', 'FPS']
        wrt = csv.DictWriter(f, head)

        wrt.writerow({head[0]: head[0], head[1]: head[1], head[2]: head[2], head[3]: head[3], head[4]: head[4],
                      head[5]: head[5]})
        wrt.writerow({
            head[0]: 'Male 22',
            head[1]: '{:.2f}bpm'.format(gt_new.mean()),
            head[2]: '{:.2f}bpm'.format(pred.mean()),
            head[3]: '{:.2f}bpm'.format(error_avg),
            head[4]: '{:.2f}%'.format(precent_deviation.mean()),
            head[5]: '{:.2f}'.format(fps.mean()),
        })

        for i in range(len(pred)):
            wrt.writerow({
                head[0]: i,
                head[1]: '{:.2f}'.format(gt_new[i]),
                head[2]: '{:.2f}'.format(pred[i]),
                head[3]: '{:.2f}'.format(error[i]),
                head[4]: '{:.2f}'.format(precent_deviation[i]),
                head[5]: '{:.2f}'.format(fps[i])
            })
