"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import cv2
import numpy as np
from openvino.inference_engine import InferQueue

from gesture_recognition_demo.common import IEModel


class PersonDetector(IEModel):
    """ Class that allows worknig with person detectpr models. """

    def __init__(self, model_path, device, ie_core, jobs, output_shape=None):
        """Constructor"""

        super().__init__(model_path, device, ie_core, 'Person Detection', output_shape)

        self.infer_queue = InferQueue(self.exec_net, jobs)
        self.infer_queue.set_infer_callback(self._process_output)

        _, _, h, w = self.input_size
        self.input_height = h
        self.input_width = w

        self.last_scales = None
        self.last_sizes = None

    def _prepare_frame(self, frame):
        """Converts input image according model requirements"""

        initial_h, initial_w = frame.shape[:2]
        scale_h, scale_w = initial_h / float(self.input_height), initial_w / float(self.input_width)

        in_frame = cv2.resize(frame, (self.input_width, self.input_height))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape(self.input_size)

        return in_frame, initial_h, initial_w, scale_h, scale_w

    def _process_output(self, request, status, userdata):
        """Converts network output to the internal format"""

        scale_h, scale_w = self.last_scales
        result = request.get_blob(self.output_name).buffer
        if result.shape[-1] == 5:  # format: [xmin, ymin, xmax, ymax, conf]
            userdata['result'] = np.array([[scale_w, scale_h, scale_w, scale_h, 1.0]]) * result
        else:  # format: [image_id, label, conf, xmin, ymin, xmax, ymax]
            scale_w *= self.input_width
            scale_h *= self.input_height
            out = np.array([[1.0, scale_w, scale_h, scale_w, scale_h]]) * result[0, 0, :, 2:]

            userdata['result'] = np.concatenate((out[:, 1:], out[:, 0].reshape([-1, 1])), axis=1)

    def async_infer(self, frame):
        """Requests model inference for the specified image"""

        in_frame, initial_h, initial_w, scale_h, scale_w = self._prepare_frame(frame)
        self.last_sizes = initial_h, initial_w
        self.last_scales = scale_h, scale_w

        userdata = {'result' : None}
        input_data = {self.input_name: in_frame.astype(np.float32)}
        self.infer_queue.async_infer(inputs=input_data, userdata=userdata)

    def wait_request(self, req_id):
        """Waits for the model output"""

        if self.last_scales is None or self.last_sizes is None:
            raise ValueError('Unexpected request')

        if self.infer_queue[req_id].wait(-1) == 0:
            return self.infer_queue.userdata[req_id]['result']
        else:
            return None

    def getIdleRequestId(self):
        return self.infer_queue.get_idle_request_info()['id']
