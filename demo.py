import argparse
from collections import deque
from operator import itemgetter
from threading import Thread

import cv2
import numpy as np
import torch
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose

import time
import sys
from PyQt5 import QtCore, QtWidgets, QtGui

from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import QDateTime, Qt
from PyQt5.QtChart import QDateTimeAxis, QValueAxis, QSplineSeries, QChart, QChartView
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QColor, QBrush
from PyQt5.QtWidgets import QApplication, QWidget


running = True
FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 2
FONTCOLOR = (255, 0, 0)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 2
LINETYPE = 1

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode', 'FrameSelector'
]




def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('label', help='label file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='recognition score threshold')
    parser.add_argument(
        '--average-size',
        type=int,
        default=1,
        help='number of latest clips to be averaged for prediction')
    args = parser.parse_args()
    return args

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Elderly Monitoring Platform")
        self.setObjectName("MainWindow")
        self.resize(640, 480)  #1196,710
        self.setStyleSheet("background-color: rgb(66,66,66);")
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setContentsMargins(11, 11, -1, -1)
        self.gridLayout.setSpacing(20)
        self.gridLayout.setObjectName("gridLayout")

        # 标题栏
        self.labTitle = QtWidgets.QLabel(self)
        self.labTitle.setObjectName("labMidTitle")
        self.labTitle.setText("Intelligent Elderly Monitoring Platform")
        self.labTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.labTitle.setStyleSheet("color: #ffffff; font: 100 20pt ;background-color:transparent")
        self.gridLayout.addWidget(self.labTitle, 0, 0, 1, 1)

        # 主面板
        self.mainFrame = QtWidgets.QFrame(self)
        self.mainFrame.setEnabled(True)
        self.mainFrame.setObjectName("grpMainPanel")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.mainFrame)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setSpacing(5)
        self.gridLayout_2.setObjectName("gridLayout_2")

        # 主面板中视频窗口
        self.playerView = QtWidgets.QLabel(self.mainFrame)
        self.playerView.setObjectName("playerView")
        self.playerView.setStyleSheet("border-width: 3px;border-style: solid;border-color: rgb(170, 0, 0);")
        self.gridLayout_2.addWidget(self.playerView, 0, 0, 2, 1)

        # 主面板中告警结果列表    
        self.alarmView = QtWidgets.QListView(self.mainFrame)
        self.alarmView.setObjectName("alarmView")
        self.alarmView.setStyleSheet("border-width: 3px;border-style: solid;border-color: rgb(170, 0, 0);color:rgb("
                                     "255,255,255);font: 15pt;")
        self.gridLayout_2.addWidget(self.alarmView, 0, 1, 1, 1)       

        self.gridLayout_2.setColumnStretch(0, 6)
        self.gridLayout_2.setColumnStretch(1, 4)
        self.gridLayout.addWidget(self.mainFrame, 1, 0, 1, 1)

        self.alarmItemModel = QStandardItemModel()
        self.alarmView.setModel(self.alarmItemModel)
        # self.setWindowFlags(Qt.FramelessWindowHint)
        self.result_queue = deque(maxlen=1)
        self.runing = True

    #   True: insert   False: replace
    def alarm(self, msg, insert_flag=True):

        item = QStandardItem(msg)
              
        if insert_flag == False: self.alarmItemModel.removeRows(0,1) 
        
        self.alarmItemModel.insertRow(0, item)            
        self.alarmView.setModel(self.alarmItemModel)

    def keyPressEvent(self, event):
        """按ESC键程序关闭界面窗口"""
        if event.key() == QtCore.Qt.Key_Escape:
            time.sleep(0.1)
            self.close()


    def play(self):
        print('Press "Esc", "q" or "Q" to exit')
        # frame_counter = 0
        text_info = {}

        start = time.time()
        while running:
            msg = 'Waiting for action ...'
            
            ret, frame = camera.read()
            frame_queue.append(np.array(frame[:, :, ::-1]))

            res_info = ''

            if len(self.result_queue) != 0:
                text_info = {}
                results = self.result_queue[0]
                for i, result in enumerate(results):
                    selected_label, score = result
                    if score < threshold:
                        break       
                    location = (0, 40 + i * 20)
                    text = selected_label + ': ' + str(round(score, 2))
                    # print('*******result:',text)
                    text_info[location] = text
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,  #FONTFACE
                                FONTCOLOR, THICKNESS, LINETYPE)
                    res_info = text

            elif len(text_info):
                for location, text in text_info.items():
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)
                    res_info = text

            else:
                cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                            THICKNESS, LINETYPE)


            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame_row, frame_col, frame_depth = frame.shape
            image = QtGui.QImage(frame.data, frame_col, frame_row, frame_col * frame_depth, QtGui.QImage.Format_RGB888)
            self.playerView.setPixmap(QtGui.QPixmap.fromImage(image))
            self.playerView.setScaledContents(True)

      
  
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break

    def text_show(self):
        text_info = {}
        pre = time.time()       


        start = time.time()
        laststate = None
        curstate = None

        norm_start = time.time()   

        while running:
            res_info = ''
            cur = time.time()
            if cur - pre < 1:
                time.sleep(0.01)
                continue
            pre = cur
            if len(self.result_queue) != 0:
                text_info = {}
                results = self.result_queue.popleft()
                for i, result in enumerate(results):
                    selected_label, score = result
                    if score < threshold:
                        break       
                    location = (0, 40 + i * 20)
                    text = selected_label + ': ' + str(round(score, 2))
                    text_info[location] = text
                    res_info = text
            elif len(text_info):
                for location, text in text_info.items():
                    res_info = text
            else: 
                res_info = ''

            if res_info != '':
                print("########res_info:",res_info)
                k = 0
                for i in range(7):
                    if res_info.split(":")[0] == label_dir[i]:
                        k = i
                        res_info = out_dir[k]
                        break 

                curstate = res_info
                et = 0
                tag = False
                if curstate == laststate and laststate != '':
                    et = time.time() - start
                    if et < 0.8: self.alarmItemModel.removeRows(0,1) 
                    tag = False

                else :                    
                    start = time.time() 
                    tag = True 
                    et = 0.1

                alarm_info = time.strftime("%m-%d %H:%M:%S", time.localtime()) + '\t\t' + curstate + '\t' + 'During: %.2fs' % et
                self.alarm(alarm_info, tag) 

                laststate = curstate
                norm_start = time.time()

            else :
                norm_et = time.time() - norm_start
                if norm_et > 5.0: 
                    start = time.time()
                    laststate = ''

            # self.alarm.moveCursor(QTextCursor.End)



def inference(result_queue):
    pre = time.time()
    score_cache = deque()
    scores_sum = 0
    while running:
        cur_windows = []
        while len(cur_windows) == 0:
            if len(frame_queue) == sample_length:
                cur_windows = list(np.array(frame_queue))
                if data['img_shape'] is None:
                    data['img_shape'] = frame_queue.popleft().shape[:2]

        cur_data = data.copy()
        cur_data['imgs'] = cur_windows
        cur_data = test_pipeline(cur_data)
        cur_data = collate([cur_data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            cur_data = scatter(cur_data, [device])[0]

        with torch.no_grad():
            scores = model(return_loss=False, **cur_data)[0]

        score_cache.append(scores)
        scores_sum += scores

        if len(score_cache) == average_size:
            scores_avg = scores_sum / average_size
            num_selected_labels = min(len(label), 5)

            scores_tuples = tuple(zip(label, scores_avg))
            scores_sorted = sorted(
                scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]
            result_queue.append(results)
            scores_sum -= score_cache.popleft()

    camera.release()
    cv2.destroyAllWindows()



def main():
    global frame_queue, camera, frame, results, threshold, sample_length, \
        data, test_pipeline, model, device, average_size, label, \
        label_dir, out_dir, res_info, running

    config = 'work_dirs1/slowfast_r101_r50_4x16x1_256e_kinetics400_rgb.py'
    checkpoint =  'work_dirs1/latest.pth' 
    label = 'work_dirs1/label_map_k400.txt'    
    average_size = 5
    threshold = 0.55
    camera_id = 0
    res_info = ''
    label_dir = {
        0:'cough', 1:'falldown', 2:'headache', 3:'chestpain',
        4:'backpain', 5:'standup', 6:'sitdown'}
    out_dir = {
        0:'cough  ', 1:'falldown', 2:'headache', 3:'chestpain',
        4:'backpain', 5:'standup', 6:'sitdown'}


    device = torch.device('cuda:0')
    model = init_recognizer(config, checkpoint, device=device) 
    camera = cv2.VideoCapture(0)  #camera_id
    data = dict(img_shape=None, modality='RGB', label=-1)

    with open(label, 'r') as f:
        label = [line.strip() for line in f]  

    # prepare test pipeline from non-camera pipeline
    cfg = model.cfg    
    sample_length = 0
    pipeline = cfg.test_pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in step['type']:
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            # remove step to decode frames
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0

    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()        

    try:
        frame_queue = deque(maxlen=sample_length)  

        pr = Thread(target=inference, args=(mainWindow.result_queue, ), daemon=True)    
        pr.start()
        time.sleep(1)
        pt = Thread(target=mainWindow.text_show, daemon=True)
        pt.start()
        time.sleep(1)

        mainWindow.show()  
        mainWindow.play()


        
        # pw.join()
    except KeyboardInterrupt:
        pass
    running = False

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
