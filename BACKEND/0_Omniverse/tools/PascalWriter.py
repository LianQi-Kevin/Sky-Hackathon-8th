import omni.kit
import omni.usd
import omni.replicator.core as rep
import xml.etree.ElementTree as ET
from omni.replicator.core import Writer, AnnotatorRegistry, BackendDispatch

import os
from typing import List, Tuple


class PascalWriter(Writer):
    def __init__(self,
                 output_dir: str,
                 rgb: bool = True,
                 bounding_box_2d_tight: bool = True,
                 image_output_format: str = "jpg",
                 semantic_types: List[str] = None,
                 bbox_height_threshold: int = 25,
                 partly_occluded_threshold: float = 0.5,
                 fully_visible_threshold: float = 0.95,
                 ):
        """Create a PASCAL VOC Writer
        Args:
            output_dir: Output directory to which PASCAL VOC annotations will be saved.
            rgb: Boolean value that indicates whether the rgb annotator will be activated and the data will be written or not. Default: True.
            bounding_box_2d_tight: Boolean value that indicates whether the bounding_box_2d_tight annotator will be activated and the data will be written or not. Default: False.
            image_output_format: String that indicates the format of saved RGB images. Default: "jpg"
            semantic_types: List of semantic types to consider when filtering annotator data. Default: ["class"]
            bbox_height_threshold: The minimum valid bounding box height, in pixels. Value must be positive integers.
            partly_occluded_threshold: Minimum occlusion factor for bounding boxes to be considered partly occluded.
            fully_visible_threshold: Minimum occlusion factor for bounding boxes to be considered fully visible.
        """
        self._output_dir = output_dir
        self._backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self._frame_id = 0
        self._image_output_format = image_output_format
        self._bbox_height_threshold = bbox_height_threshold
        self._semantic_types = ["class"] if semantic_types is None else semantic_types
        self._partly_occluded_threshold = partly_occluded_threshold
        self._fully_visible_threshold = fully_visible_threshold
        self._rgb_path = "JPEGImages"
        self._VOC_path = "Annotations"

        self.annotators = []

        # RGB
        if rgb:
            self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))

        # Bounding Box 2D
        if bounding_box_2d_tight:
            self.annotators.append(AnnotatorRegistry.get_annotator("bounding_box_2d_tight",
                                                                   init_params={"semanticTypes": self._semantic_types}))

    @staticmethod
    def _check_bbox_area(bbox_data: dict, size_limit: float = 0.5) -> bool:
        length = abs(bbox_data['x_min'] - bbox_data['x_max'])
        width = abs(bbox_data['y_min'] - bbox_data['y_max'])

        area = length * width
        if area > size_limit:
            return True
        else:
            return False

    def _write_rgb(self, data, sub_dir, annotator: str = "rgb"):
        # Save the rgb data under the correct path
        self._backend.write_image(os.path.join(sub_dir, self._rgb_path, f"{self._frame_id}.png"), data[annotator])

    def _write_object_detection(self, data, sub_dir,
                                render_product_annotator: str = "",
                                bbox_2d_tight_annotator: str = "",
                                bbox_2d_loose_annotator: str = ""):
        pass

    def write(self, data):
        render_products = [k for k in data.keys() if k.startswith("rp_")]
        if len(render_products) == 1:
            sub_dir = data[render_products[0]]["camera"].split("/")[-1]
            self._write_rgb(data=data, sub_dir=sub_dir, annotator="rgb")

