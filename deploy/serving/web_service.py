# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle_serving_server.web_service import WebService, Op
import numpy as np
import sys
import base64
import cv2
import io
from preprocess_ops import Resize, NormalizeImage, Permute, PadStride, Compose


class RetinaNetOp(Op):
    """RetinaNetOp
    
    RetinaNet service op
    """

    def init_op(self):
        self.eval_transforms = Compose([
            Resize(target_size=[800, 1333]), 
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,0.225]), 
            Permute(),
            PadStride(32)
        ])

    def preprocess(self, input_dicts, data_id, log_id):
        """preprocess
        
        In preprocess stage, assembling data for process stage. users can 
        override this function for model feed features.
        Args:
            input_dicts: input data to be preprocessed
            data_id: inner unique id, increase auto
            log_id: global unique id for RTT, 0 default
        Return:
            output_data: data for process stage
            is_skip_process: skip process stage or not, False default
            prod_errcode: None default, otherwise, product errores occured.
                          It is handled in the same way as exception. 
            prod_errinfo: "" default
        """
        (_, input_dict), = input_dicts.items()
        batch_size = len(input_dict.keys())
        imgs = []
        imgs_info = {'im_shape':[], 'scale_factor':[]}
        for key in input_dict.keys():
            data = base64.b64decode(input_dict[key].encode('utf8'))
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_info = {
                'scale_factor': np.array([1., 1.], dtype=np.float32),
                'im_shape': img.shape[:2],
            }
            img, im_info = self.eval_transforms(img, im_info)
            imgs.append(img[np.newaxis, :].copy())
            imgs_info["im_shape"].append(im_info["im_shape"][np.newaxis, :].copy())
            imgs_info["scale_factor"].append(im_info["scale_factor"][np.newaxis, :].copy())
            
        input_imgs = np.concatenate(imgs, axis=0)
        input_im_shape = np.concatenate(imgs_info["im_shape"], axis=0)
        input_scale_factor = np.concatenate(imgs_info["scale_factor"], axis=0)
        return {"image": input_imgs, "im_shape": input_im_shape, "scale_factor": input_scale_factor}, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        """postprocess
        In postprocess stage, assemble data for next op or output.
        Args:
            input_data: data returned in preprocess stage, dict(for single predict) or list(for batch predict)
            fetch_data: data returned in process stage, dict(for single predict) or list(for batch predict)
            data_id: inner unique id, increase auto
            log_id: logid, 0 default
        Returns: 
            fetch_dict: fetch result must be dict type.
            prod_errcode: None default, otherwise, product errores occured.
                          It is handled in the same way as exception.
            prod_errinfo: "" default
        """

        np_boxes = list(fetch_dict.values())[0]
        keep = (np_boxes[:, 1] > 0.5) & (np_boxes[:, 0] > -1)
        np_boxes = np_boxes[keep, :]
        result = {"class_id": [], "confidence": [], "left_top": [], "right_bottom": []}
        for dt in np_boxes:
            clsid, bbox, score = int(dt[0]), dt[2:], dt[1]

            xmin, ymin, xmax, ymax = bbox

            result["class_id"].append(clsid)
            result["confidence"].append(score)
            result["left_top"].append([xmin, ymin])
            result["right_bottom"].append([xmax, ymax])

        result["class_id"] = str(result["class_id"])
        result["confidence"] = str(result["confidence"])
        result["left_top"] = str(result["left_top"])
        result["right_bottom"] = str(result["right_bottom"])

        return result, None, ""


class RetinaNetService(WebService):
    """RetinaNetService
    
    RetinaNet service class.
    """

    def get_pipeline_response(self, read_op):
        retinanet_op = RetinaNetOp(name="retinanet", input_ops=[read_op])
        return retinanet_op


# define the service class
uci_service = RetinaNetService(name="retinanet")
# load config and prepare the service
uci_service.prepare_pipeline_config("deploy/serving/config.yml")
# start the service
uci_service.run_service()