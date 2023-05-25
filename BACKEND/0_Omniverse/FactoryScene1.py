"""
对 example.py 做一些修改以实现更多的物体变动和镜头移动，用于Sky-Hackathon-8th

此处补充一些omni.replicator.core的注意事项
1. rep.distribution.uniform(lower, upper)
    lower 的每一个值都必须要小于 upper
"""

import omni.replicator.core as rep
import os
import glob
import random
from typing import List, Tuple


def get_num_URLS(num=4, filename_path=os.path.join("Assets/BoxURLS", "*.txt"),
                 semantics: List[Tuple[str, str]] = None) -> List[rep.create.ReplicatorItem]:
    if semantics is None:
        semantics = [('class', 'box')]
    urls = []
    for url_path in glob.glob(filename_path):
        with open(url_path, "r") as url_f:
            urls += [i[:-2] if i.endswith('\n') else i for i in url_f.readlines()]
    models = []
    for url in random.sample(urls, num):
        models.append(rep.create.from_usd(url, semantics=semantics))
    return models


# setup random view range for camera: low point, high point
sequential_pos = [(-800, 220, -271), (800, 220, 500)]

# position of look-at target
look_at_position = [(-430, 78, 10), (355, 78, 90)]

# setup working layer
with rep.new_layer():
    # Basic scenario
    # 无头模式下场景无默认光源，添加额外光源(参照默认环境光设置)
    rep.create.light(light_type="Distant", temperature=6500, intensity=3000,
                     rotation=(315, 0, 0), scale=1, name="DefaultLight")
    WORKSHOP = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Industrial/Buildings/Warehouse/Warehouse01.usd'
    CONVEYOR = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Conveyors/ConveyorBelt_A/ConveyorBelt_A23_PR_NVD_01.usd'
    workshop = rep.create.from_usd(WORKSHOP)
    conveyor1 = rep.create.from_usd(CONVEYOR)
    conveyor2 = rep.create.from_usd(CONVEYOR)
    with workshop:
        rep.modify.pose(
            position=(0, 0, 0),
            rotation=(0, -90, -90)
        )
    with conveyor1:
        rep.modify.pose(
            position=(-40, 0, 0),
            rotation=(0, -90, -90)
        )
    with conveyor2:
        rep.modify.pose(
            position=(-40, 0, 100),
            rotation=(-90, 90, 0)
        )

    # boxs
    CARTON_URLS = get_num_URLS(num=4, semantics=[('class', 'box')])
    box1, box2, box3, box4 = CARTON_URLS[0], CARTON_URLS[1], CARTON_URLS[2], CARTON_URLS[3]

    with box1:
        rep.modify.pose(
            position=(-350, 78, 57),
            rotation=(0, -90, -90),
        )
    with box2:
        rep.modify.pose(
            position=(-100, 78, 57),
            rotation=(0, -90, -90),
        )
    with box3:
        rep.modify.pose(
            position=(100, 78, 57),
            rotation=(0, -90, -90),
        )
    with box4:
        rep.modify.pose(
            position=(200, 78, 57),
            rotation=(0, -90, -90),
        )

    # define lighting function
    def sphere_lights(num):
        lights = rep.create.light(
            light_type="Sphere",
            temperature=rep.distribution.normal(3500, 500),
            intensity=rep.distribution.normal(15000, 5000),
            position=rep.distribution.uniform((-300, -300, -300), (300, 300, 300)),
            scale=rep.distribution.uniform(50, 100),
            count=num
        )
        return lights.node
    rep.randomizer.register(sphere_lights)

    # define function to create random position range for target
    def get_shapes():
        # 选定所有键值对为("class", "box")的对象
        shapes = rep.get.prims(semantics=[('class', 'box')])
        # 针对选定的所有对象批量配置
        with shapes:
            rep.modify.pose(
                position=rep.distribution.uniform((0, -50, 0), (0, 50, 0)),
            )
        return shapes.node
    rep.randomizer.register(get_shapes)

    # 创建相机并配置渲染
    camera = rep.create.camera(position=sequential_pos[0], look_at=look_at_position[0])
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))

    with rep.trigger.on_frame(num_frames=50):  # number of picture
        rep.randomizer.sphere_lights(4)  # number of lighting source
        rep.randomizer.get_shapes()
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform(sequential_pos[0], sequential_pos[1]),
                look_at=rep.distribution.uniform(look_at_position[0], look_at_position[1]))

    # Initialize and attach writer for Kitti format data
    writer = rep.WriterRegistry.get("KittiWriter")
    writer.initialize(
        output_dir=os.path.join(os.getcwd(), "_output"),
        bbox_height_threshold=10,
        fully_visible_threshold=0.95,
        omit_semantic_type=True
    )
    writer.attach([render_product])

    # required for headless mode
    rep.orchestrator.preview()
