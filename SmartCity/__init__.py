import os
import time
from threading import Thread

if os.environ.get("SERVER", False):
    from SmartCity.settings import buffer_queue, resultQueue
    from utils import run
    from utils.ArgumentsHandler import argHandler
    import tensorflow as tf

    def worker_process(worker_id, FLAGS):
        tfnet = run.Initialize(FLAGS)
        print("Worker {} Ready".format(worker_id))
        while True:
            try:
                counter, frame = buffer_queue.get()
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
                        tfnet.framework.postprocess(single_out, img, resultQueue, counter=counter)

            except KeyboardInterrupt:
                print("Keyboard Interrupt. Exiting Worker {}".format(worker_id))
                break
            except Exception as e:
                print("Worker Waiting " + str(e))
                time.sleep(2)


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
