


from multiprocessing import Process, Queue
from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_result)

import pickle as pkl
import cv2
import numpy as np
import os
import streamlit as st
import time
from llama_cpp import Llama



### FUNCTIONS ###

class QueryWorker(Process):

    def __init__(self, queue, resultsFromChildren, num):
        super().__init__()

        self.screenshotPath = "extracted-screenshots"
        self.filePath = "extracted-files"
        self.queue = queue
        self.results = resultsFromChildren
        self.model = init_pose_model(
            "mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark.py",
            "model_paths/pose_check.pth", device="cuda")
        self.name = num

        self.is_model = True

    def run(self):
        for data in iter(self.queue.get, None):
            self.processWork(data)
        print("DONE")

    def processWork(self, entity):
        results = []

        if self.is_model:
            self.is_model = False
            self.model = init_pose_model(
                "mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark.py",
                "model_paths/pose_check.pth", device="cuda")



        for frame in entity[0]:
            pose_results, returned_outputs = inference_top_down_pose_model(
                self.model,
                frame,
                bbox_thr=None,
                format='xyxy',
                dataset=None,
                dataset_info=None,
                return_heatmap=False,
                outputs=None)

            img = vis_pose_result(
                self.model,
                frame,
                pose_results
            )

            cv2.imwrite(f"extracted-files/vis/{self.name}-{entity[1]}.png", img)

            pose_results = pose_results[0]
            results.append(pose_results["keypoints"])


        results = np.array(results)
        with open(os.path.join("extracted-files/poses", f'{entity[2]}-{entity[1]}.pkl'), 'wb') as f:
            pkl.dump(results, f)


class WorkScheduler():

    def __init__(self, config):
        self.numWorkers = config["numWorkers"]
        self.workerPool = []

        self.resultsFromChildren = Queue()
        self.workToDo = Queue()

        print("Setting up workers...")
        for worker in range(self.numWorkers):
            self.workerPool.append(QueryWorker(self.workToDo, self.resultsFromChildren, str(worker)))

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

if __name__ == '__main__':


#### STREAMLIT LAYOUT ####

    st.title("RT-ASL Demo")
    st.header("Webcam Live Stream")
    web_cam = st.image(np.ones([1440, 1920])*0.5)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.write(' ')
    with c2:
        start_capture = st.button('Start Capture')
    with c3:
        st.write(' ')
    with c4:
        end_capture = st.button('End Capture')
    with c5:
        st.write(' ')


    # Code required for
    if start_capture:
        capture = cv2.VideoCapture(0)
        config = {"numWorkers": 1}
        workerScheduler = WorkScheduler(config=config)
        idx = 0
        while True:
            ret, frame = capture.read()
            web_cam.image(frame)
            workerScheduler.add_work([[frame], idx, 0])
            idx += 1


    if end_capture:
        from os import listdir
        from os.path import isfile, join

        time.sleep(1)

        mypath = "extracted-files/vis"

        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        worker_1, worker_2 = [], []
        for file in onlyfiles:
            if file[0] == "0":
                worker_1.append(file)


        processes = st.container()
        with processes:
            st.header("Parsed Images")
            c2, c4= st.columns(2)
            with c2:
                disp_1 = st.image(np.ones([1440, 1920])*0.5)
                st.text("Process 1")

        st.header("Langauge Parsing")
        prediction_chat = st.container(height=300)

        hold_over = max(len(worker_1), len(worker_2))

        idx_1 = 0
        idx_2 = 0
        for _ in range(0, hold_over):
            if len(worker_1)-1  < idx_1:
                idx_1 = 0
            if len(worker_2)-1 < idx_2:
                idx_2 = 0

            disp_1.image(cv2.imread(f"extracted-files/vis/{worker_1[idx_1]}"))

            idx_1 += 1
            idx_2 += 1

            time.sleep(0.1)


        class GLOFE():
            def __init__(self, path):
                self.message = "you name"
                self.llm = Llama(model_path="model_paths/llama-2-7b.Q3_K_S.gguf")
                self.prompt = """You are going to be given a sentence which may be malformed. 
Given the conversation context, return the sentence with corrected grammar and makes the most sense:

Sentence #1: Hi it is nice to meet you, im John.
"""

            def getResult(self):
                time.sleep(0.75)
                return self.message

            def getLLMResult(self):
                updated_prompt = self.prompt + "\nSentence #2: "+ self.message + "? \nUpdated Sentence #2: "
                print(updated_prompt)
                result = self.llm(updated_prompt)
                return result["choices"][0]["text"]

        model = GLOFE("extracted-files/poses")
        prediction_chat.chat_message("ai").text(f"GLOFE prediction: {model.getResult()}")
        prediction_chat.chat_message("human").text(f"LLM improved: {model.getLLMResult()}")



