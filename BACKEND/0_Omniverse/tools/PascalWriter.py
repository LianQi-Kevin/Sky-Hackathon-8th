"""
Custom Writer for PASCAL VOC

Reference:
* https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator/custom_writer.html
* https://arleyzhang.github.io/articles/1dc20586/
"""

import numpy as np
from omni.replicator.core import BackendDispatch, Writer, WriterRegistry
from omni.syntheticdata.scripts.SyntheticData import SyntheticData

import io
import os
from typing import List
import xml.etree.ElementTree as ET


class PascalWriter(Writer):
    def __init__(self,
                 output_dir: str,
                 semantic_types: List[str] = None,
                 bbox_height_threshold: int = 25,
                 fully_visible_threshold: float = 0.95,
                 partly_occluded_threshold: float = 0.15
                 ):
        """Create a PASCAL VOC Writer
        Args:
            output_dir: Output directory to which PASCAL VOC annotations will be saved.
            semantic_types: List of semantic types to consider when filtering annotator data. Default: ["class"]
            bbox_height_threshold: The minimum valid bounding box height, in pixels. Value must be positive integers.
            fully_visible_threshold: Minimum occlusion factor for bounding boxes to be considered fully visible.
            partly_occluded_threshold: Minimum occlusion factor for bounding boxes to be considered partly occluded.
        """
        self._output_dir = output_dir
        self._backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self._frame_id = 0
        self._image_output_format = "jpg"
        self._bbox_height_threshold = bbox_height_threshold
        self._semantic_types = ["class"] if semantic_types is None else semantic_types
        self._partly_occluded_threshold = partly_occluded_threshold
        self._fully_visible_threshold = fully_visible_threshold
        self._rgb_path = "JPEGImages"
        self._VOC_path = "Annotations"
        self._EPS = 1e-5

        self.annotators = [
            "rgb",
            "bounding_box_2d_tight_fast",
            "bounding_box_2d_loose_fast",
            "camera_params",
        ]
        semantic_filter_predicate = ":*; ".join(self._semantic_types) + ":*"
        SyntheticData.Get().set_instance_mapping_semantic_filter(semantic_filter_predicate)

    @staticmethod
    def _create_element(name: str, item=None) -> ET.Element:
        element = ET.Element(name)
        element.text = str(item)
        return element

    def _indent(self, elem, level=0):
        """
        add '\n' in xml file
        """
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def _create_PASCAL_VOC(self, data: dict, format_: bool = True) -> ET.ElementTree:
        # verify
        assert "filename" in data.keys(), f"Not found filename tag in {data}"
        assert "object" in data.keys(), f"Not found object tag in {data}"
        assert len(data["object"]) != 0, f"Object is empty in {data}"
        # base tag
        root = ET.Element("annotation")
        root.append(self._create_element("folder", data.get("folder", "VOC2007")))
        root.append(self._create_element("filename", data.get("filename")))
        root.append(self._create_element("segmented", data.get("segmented", 0)))
        # img size
        size = ET.Element("size")
        size.append(self._create_element("width", data["size"].get("width", 512) if "size" in data.keys() else 512))
        size.append(self._create_element("height", data["size"].get("height", 512) if "size" in data.keys() else 512))
        size.append(self._create_element("depth", data["size"].get("depth", 512) if "size" in data.keys() else 3))
        root.append(size)
        # object
        for object_ in data["object"]:
            label = ET.Element("object")
            label.append(self._create_element("name", object_.get("name")))
            label.append(self._create_element("truncated", object_.get("truncated")))
            label.append(self._create_element("difficult", object_.get("difficult")))
            bndbox = ET.Element("bndbox")
            bndbox.append(self._create_element("xmin", object_["bndbox"].get("xmin")))
            bndbox.append(self._create_element("ymin", object_["bndbox"].get("ymin")))
            bndbox.append(self._create_element("xmax", object_["bndbox"].get("xmax")))
            bndbox.append(self._create_element("ymax", object_["bndbox"].get("ymax")))
            label.append(bndbox)
            root.append(label)
        # write
        if format_:
            self._indent(root)
        return ET.ElementTree(root)

    def _write_rgb(self, data, sub_dir: str = None, annotator: str = "rgb"):
        # Save the rgb data under the correct path
        self._backend.write_image(os.path.join("" if sub_dir is None else sub_dir,
                                               self._rgb_path, f"{self._frame_id}.{self._image_output_format}"),
                                  data[annotator])

    def _write_object_detection(self, data, sub_dir: str = None, render_product_annotator: str = None,
                                bbox_2d_tight_annotator: str = "bounding_box_2d_tight_fast",
                                bbox_2d_loose_annotator: str = "bounding_box_2d_loose_fast"):

        rp_width = data[render_product_annotator]["resolution"][0]
        rp_height = data[render_product_annotator]["resolution"][1]

        bbox_tight = data[bbox_2d_tight_annotator]["data"]
        bbox_loose = data[bbox_2d_loose_annotator]["data"]

        bbox_tight_bbox_ids = data[bbox_2d_tight_annotator]["info"]["bboxIds"]
        bbox_loose_bbox_ids = data[bbox_2d_loose_annotator]["info"]["bboxIds"]

        # For box in tight, find the corresponding index of box in loose
        bbox_loose_indices = np.where(np.isin(bbox_loose_bbox_ids, bbox_tight_bbox_ids))[0]
        selected_bbox_loose = bbox_loose[bbox_loose_indices]

        labels_dict = {
            "folder": self._output_dir,
            "filename": f"{self._frame_id}.{self._image_output_format}",
            "segmented": 0,
            "size": {
                "width": rp_width,
                "height": rp_height,
                "depth": 3
            },
            "object": []
        }

        for box_tight, box_loose in zip(bbox_tight, selected_bbox_loose):
            label = {
                "name": None,
                "bndbox": {}
            }

            # Skip boxes shorter than threshold pixels in height
            if box_tight["y_max"] - box_tight["y_min"] < self._bbox_height_threshold:
                continue

            area_tight = (box_tight["x_max"] - box_tight["x_min"]) * (box_tight["y_max"] - box_tight["y_min"])
            area_loose = (box_loose["x_max"] - box_loose["x_min"]) * (box_loose["y_max"] - box_loose["y_min"])
            area_ratio = area_tight / (area_loose + self._EPS)

            if area_ratio >= self._fully_visible_threshold:
                pass
            elif area_ratio >= 1 - self._partly_occluded_threshold:
                label["truncated"] = 1
            else:
                label["truncated"] = 1
                label["difficult"] = 1

            # Check if bounding boxes are in the viewport
            if (
                box_tight["x_min"] < 0
                or box_tight["y_min"] < 0
                or box_tight["x_max"] > rp_width
                or box_tight["y_max"] > rp_height
                or box_tight["x_min"] > rp_width
                or box_tight["y_min"] > rp_height
                or box_tight["y_max"] < 0
                or box_tight["x_max"] < 0
            ):
                continue

            label["name"] = data[bbox_2d_tight_annotator]["info"]["idToLabels"].get(
                box_tight["semanticId"]).get("class", "Unlabelled")
            label["bndbox"]["xmin"] = box_tight["x_min"]
            label["bndbox"]["ymin"] = box_tight["y_min"]
            label["bndbox"]["xmax"] = box_tight["x_max"]
            label["bndbox"]["ymax"] = box_tight["y_max"]

            labels_dict["object"].append(label)
        buf = io.BytesIO()
        self._create_PASCAL_VOC(data=labels_dict, format_=True).write(buf, encoding='utf-8', xml_declaration=True)
        self._backend.write_blob(os.path.join("" if sub_dir is None else sub_dir,
                                              self._VOC_path, f"{self._frame_id}.xml"), buf.getvalue())

    def write(self, data):
        render_products = [k for k in data.keys() if k.startswith("rp_")]
        if len(render_products) == 1:
            self._write_rgb(data=data, sub_dir=None, annotator="rgb")
            self._write_object_detection(
                data=data, sub_dir=None, render_product_annotator=render_products[0],
                bbox_2d_tight_annotator="bounding_box_2d_tight_fast",
                bbox_2d_loose_annotator="bounding_box_2d_loose_fast"
            )
        else:
            for render_product in render_products:
                pass
        self._frame_id += 1


WriterRegistry.register(PascalWriter)
