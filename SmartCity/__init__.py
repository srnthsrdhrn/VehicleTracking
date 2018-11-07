import os
import time
from queue import Queue
from threading import Thread
import django

django.setup()

if os.environ.get("SERVER", False):
    buffer_queue = Queue()
    resultQueue = Queue()
    from custom_utils import run
    from custom_utils.ArgumentsHandler import argHandler
    from SmartCity.settings import resultQueueDict, bufferQueueDict
    import tensorflow as tf


    def worker_process(worker_id, FLAGS):
        global buffer_queue, resultQueue
        tfnet = run.Initialize(FLAGS)
        print("Worker {} Ready".format(worker_id))
        while True:
            try:
                counter, frame, video_id = buffer_queue.get()
                with tf.device("/cpu:0"):
                    preprocessed = tfnet.framework.preprocess(frame)

                    buffer_inp = list()
                    buffer_pre = list()
                    buffer_inp.append(frame)
                    buffer_pre.append(preprocessed)

                    feed_dict = {tfnet.inp: buffer_pre}
                with tf.device("/gpu:0"):
                    net_out = tfnet.sess.run(tfnet.out, feed_dict)

                with tf.device("/cpu:0"):
                    for img, single_out in zip(buffer_inp, net_out):
                        tfnet.framework.postprocess(single_out, img, resultQueue, video_id, counter=counter)

            except KeyboardInterrupt:
                print("Keyboard Interrupt. Exiting Worker {}".format(worker_id))
                break
            except Exception as e:
                print("Worker Waiting " + str(e))
                time.sleep(2)


    def round_robin_enqueue(bufferQueueDict, buffer_queue):
        while True:
            try:
                for key, value in bufferQueueDict.items():
                    counter, frame = value.get()
                    buffer_queue.put([counter, frame, key])
            except Exception as e:
                print(str(e))


    def round_robin_dequeue(resultQueueDict, resultQueue):
        while True:
            try:
                counter, detections, boxes_final, imgcv, video_id = resultQueue.get()
                resultQueueDict[video_id].put([counter, detections, boxes_final, imgcv])
            except Exception as e:
                print(str(e))


    FLAGS = run.manual_setting()
    FLAG = argHandler()
    FLAG.setDefaults()
    FLAG.update(FLAGS)
    for id in range(0, FLAGS.number_of_parallel_threads):
        try:
            t = Thread(target=worker_process, args=(id + 1, FLAGS))
            t.daemon = True
            t.start()

        except Exception as e:
            pass
    t = Thread(target=round_robin_enqueue, args=(settings.bufferQueueDict, buffer_queue))
    t.daemon = True
    t.start()
    t = Thread(target=round_robin_dequeue, args=(settings.resultQueueDict, resultQueue))
    t.daemon = True
    t.start()
