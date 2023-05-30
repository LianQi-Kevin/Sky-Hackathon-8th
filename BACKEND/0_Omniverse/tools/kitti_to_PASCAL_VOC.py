import os
import xml.etree.ElementTree as ET


def __create_element(name: str, item=None) -> ET.Element:
    element = ET.Element(name)
    element.text = str(item)
    return element


def __indent(elem, level=0):
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
            __indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def kitti_decode(filepath: str, img_suffix: str = ".jpg", folder: str = "VOC2007", img_size: tuple = (512, 512, 3)) -> dict:
    assert os.path.exists(filepath), f"{filepath} not exists"
    # get kitti bndboxs
    with open(filepath, "r", encoding="UTF-8") as f:
        bndboxs = [i[: -1].split(" ") if i.endswith("\n") else i.split(" ") for i in f.readlines()]
    return {
        "folder": folder,
        "filename": f"{os.path.splitext(os.path.basename(filepath))[0]}{img_suffix}",
        "size": {
            "width": img_size[0],
            "height": img_size[1],
            "depth": img_size[2]
        },
        "segmented": 0,
        "object": [{
            "name": bndbox[0],
            "truncated": 0 if int(bndbox[2]) == 0 else 1,
            "difficult": 0 if int(bndbox[2]) == 0 else 1,
            "bndbox": {
                "xmin": bndbox[4],
                "ymin": bndbox[5],
                "xmax": bndbox[6],
                "ymax": bndbox[7]
            }
        } for bndbox in bndboxs]
    }


def create_PASCAL_VOC(data: dict, format_: bool = False, img_size: tuple = (512, 512, 3)) -> ET.ElementTree:
    # verify
    assert "filename" in data.keys(), f"Not found filename tag in {data}"
    assert "object" in data.keys(), f"Not found object tag in {data}"
    assert len(data["object"]) != 0, f"Object is empty in {data}"
    # base tag
    root = ET.Element("annotation")
    root.append(__create_element("folder", data.get("folder", "VOC2007")))
    root.append(__create_element("filename", data.get("filename")))
    root.append(__create_element("segmented", data.get("segmented", 0)))
    # img size
    size = ET.Element("size")
    size.append(__create_element("width", data["size"].get("width", img_size[0]) if "size" in data.keys() else img_size[0]))
    size.append(__create_element("height", data["size"].get("height", img_size[1]) if "size" in data.keys() else img_size[1]))
    size.append(__create_element("depth", data["size"].get("depth", img_size[2]) if "size" in data.keys() else img_size[2]))
    root.append(size)
    # object
    for object_ in data["object"]:
        label = ET.Element("object")
        label.append(__create_element("name", object_.get("name")))
        label.append(__create_element("truncated", object_.get("truncated")))
        label.append(__create_element("difficult", object_.get("difficult")))
        bndbox = ET.Element("bndbox")
        bndbox.append(__create_element("xmin", object_["bndbox"].get("xmin")))
        bndbox.append(__create_element("ymin", object_["bndbox"].get("ymin")))
        bndbox.append(__create_element("xmax", object_["bndbox"].get("xmax")))
        bndbox.append(__create_element("ymax", object_["bndbox"].get("ymax")))
        label.append(bndbox)
        root.append(label)
    # write
    if format_:
        __indent(root)
    return ET.ElementTree(root)


if __name__ == '__main__':
    data_ = kitti_decode(
        filepath="../_output_2023-05-28-16-45/Camera/object_detection/2.txt",
        img_suffix=".png",
        img_size=(1024, 1024, 4),
        folder="Sky-Hackathon-8th"
    )
    create_PASCAL_VOC(data_, True).write("2.xml", encoding='utf-8', xml_declaration=True)
