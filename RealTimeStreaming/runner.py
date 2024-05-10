from multiprocessing import Process, Queue
from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo
import pickle as pkl
import cv2
import numpy as np
import os
import keyboard

class QueryWorker(Process):

    def __init__(self, queue, resultsFromChildren):
        super().__init__()

        self.screenshotPath = "extracted-screenshots"
        self.filePath = "extracted-files"
        self.queue = queue
        self.results = resultsFromChildren
        self.pose_model = init_pose_model("mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark.py", "model_paths/pose_check.pth",device="cuda")

        self.dataset = self.pose_model.cfg.data['test']['type']
        self.dataset_info = DatasetInfo(self.pose_model.cfg.data['test'].get('dataset_info', None))
    def run(self):
        for data in iter(self.queue.get, None):
            self.processWork(data)
        print("DONE")

    def processWork(self, entity):
        results = []

        for frame in entity[0]:
            pose_results, returned_outputs = inference_top_down_pose_model(
                self.pose_model,
                frame,
                bbox_thr=None,
                format='xyxy',
                dataset=self.dataset,
                dataset_info=self.dataset_info,
                return_heatmap=False,
                outputs=None)


            pose_results = pose_results[0]
            results.append(pose_results["keypoints"])

        results = np.array(results)
        with open(os.path.join("extracted-files/predictions", f'{entity[2]}-{entity[1]}.pkl'), 'wb') as f:
            pkl.dump(results, f)


class WorkScheduler():

    def __init__(self, config):
        self.numWorkers = config["numWorkers"]
        self.workerPool = []

        self.resultsFromChildren = Queue()
        self.workToDo = Queue()

        print("Setting up workers...")
        for worker in range(self.numWorkers):
            self.workerPool.append(QueryWorker(self.workToDo, self.resultsFromChildren))

        for worker in range(self.numWorkers):
            self.workerPool[worker].start()

    def add_work(self, work):
        self.workToDo.put(work)

    def get_results(self):
        for worker in range(self.numWorkers):
            self.workToDo.put(None)

        for worker in range(self.numWorkers):
            self.workerPool[worker].join()

        combined_results = {}

        return combined_results

batch_size = 1

if __name__ == '__main__':

    config = {"numWorkers": 2}

    workerScheduler = WorkScheduler(config=config)

    capture = cv2.VideoCapture(0)
    end = 200

    capture.set(1, 0)
    current_sentance = 0
    is_pressed = False
    for idx in range(0, end):

        if keyboard.is_pressed("p"):
            if not is_pressed:
                is_pressed=True
                current_sentance +=1
        else:
            is_pressed = False

        frames = []
        for _ in range(0, batch_size):
            ret, frame = capture.read()
            frames.append(frame)

        workerScheduler.add_work([frames, idx, current_sentance])
        print(idx)

    capture.release()
    results = workerScheduler.get_results()

    exit(0)