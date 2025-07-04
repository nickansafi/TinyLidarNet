import time
import numpy as np
from numba import njit
from f1tenth_benchmarks.utils.BasePlanner import BasePlanner
import tensorflow as tf

class Test(BasePlanner):
    def __init__(self, test_id, skip_n, pre, model_path):
        super().__init__("TinyLidarNet", test_id)
        self.pre = pre
        self.skip_n = skip_n
        self.model_path = model_path
        self.name = 'TinyLidarNet'
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_details = self.interpreter.get_output_details()
        self.scan_buffer = np.zeros((2, 20))

        self.temp_scan = []

        self.scans = [[] for i in range(9)]

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

    def render_waypoints(self, *args, **kwargs):
        pass
        
    def transform_obs(self, scan):
        self.scan_buffer
        scan = scan[:1080]
        scan = scan[::54]
        if self.scan_buffer.all() ==0: # first reading
            for i in range(self.scan_buffer.shape[0]):
                self.scan_buffer[i, :] = scan
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan

        scans = np.reshape(self.scan_buffer, (-1))
        return scans

    
    def plan(self, obs):
        scans = obs['scan'][::2]

        noise = np.random.normal(0, 0.5, scans.shape)
        scans = scans + noise
        
        scans = np.array(scans)
        scans[scans>10] = 10

        self.scans.pop(0)
        self.scans.append(scans)
        if self.scans[0] == []:
            return 0.0, 0.0
        times = [[i*0.025]*len(scans) for i in range(5)]
        scans = np.stack([self.scans[::2], times], axis=-1)
        scans = np.expand_dims(scans, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_index, scans)
        
        start_time = time.time()
        self.interpreter.invoke()
        inf_time = time.time() - start_time
        inf_time = inf_time*1000
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        steer = output[0,0]
        speed = output[0,1]
        min_speed = 1
        max_speed = 8
        speed = self.linear_map(speed, 0, 1, min_speed, max_speed) 
        action = np.array([steer, speed])

        return action