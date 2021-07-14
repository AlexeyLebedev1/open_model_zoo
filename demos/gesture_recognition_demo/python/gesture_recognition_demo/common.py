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

import logging as log
from openvino.inference_engine import IECore, get_version


def load_ie_core(device, cpu_extension=None):
    """Loads IE Core"""
    log.info('OpenVINO Inference Engine')
    log.info('\tbuild: {}'.format(get_version()))
    ie = IECore()
    if device == "CPU" and cpu_extension:
        ie.add_extension(cpu_extension, "CPU")

    return ie


class IEModel:  # pylint: disable=too-few-public-methods
    """ Class that allows worknig with Inference Engine model. """

    def __init__(self, model_path, device, ie_core, model_type, output_shape=None):
        """Constructor"""
        if model_path.endswith((".xml", ".bin")):
            model_path = model_path[:-4]
        log.info('Reading {} model {}'.format(model_type, model_path + ".xml"))
        self.net = ie_core.read_network(model_path + ".xml", model_path + ".bin")
        assert len(self.net.input_info) == 1, "One input is expected"

        self.exec_net = ie_core.load_network(network=self.net,
                                             device_name=device)

        log.info('The {} model {} is loaded to {}'.format(model_type, model_path + ".xml", device))

        self.input_name = next(iter(self.net.input_info))
        if len(self.net.outputs) > 1:
            if output_shape is not None:
                candidates = []
                for candidate_name in self.net.outputs:
                    candidate_shape = self.exec_net.output_info[candidate_name].shape
                    if len(candidate_shape) != len(output_shape):
                        continue

                    matches = [src == trg or trg < 0
                               for src, trg in zip(candidate_shape, output_shape)]
                    if all(matches):
                        candidates.append(candidate_name)

                if len(candidates) != 1:
                    raise Exception("One output is expected")

                self.output_name = candidates[0]
            else:
                raise Exception("One output is expected")
        else:
            self.output_name = next(iter(self.net.outputs))

        self.input_size = self.net.input_info[self.input_name].input_data.shape
        self.output_size = self.exec_net.output_info[self.output_name].shape
