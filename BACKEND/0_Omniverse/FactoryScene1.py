"""
对 example.py 做一些修改以实现更多的物体变动和镜头移动，用于Sky-Hackathon-8th
"""
import os
import glob
import random
import time
from typing import List, Tuple

import omni.replicator.core as rep
from omni.replicator.core.scripts.utils import ReplicatorItem
from omni.replicator.core import WriterRegistry
from tools.PascalWriter import PascalWriter


ROOTPATH = "E:/Python/Sky-Hackathon-8th/BACKEND/0_Omniverse"
NUM_FRAMES = 50


def get_num_box_node(item_num: int = 4,
                     filename_path: str = os.path.join("Assets/BoxURLS", "*.txt"),
                     semantics: List[Tuple[str, str]] = None
                     ) -> List[ReplicatorItem]:
    """
    向 distribution_area 中添加 item_num 个物品, 并批量初始化其 rotation 和 semantics
    """
    urls = []
    for url_path in glob.glob(filename_path):
        with open(url_path, "r") as url_f:
            urls += [i[:-1] if i.endswith('\n') else i for i in url_f.readlines()]
    return [rep.create.from_usd(url, semantics=semantics) for url in random.sample(urls, item_num)]


def get_num_same_node(item_num: int, usd: str, semantics: List[Tuple[str, str]] = None) -> List[ReplicatorItem]:
    return [rep.create.from_usd(usd=usd, semantics=semantics) for _ in range(item_num)]


with rep.new_layer():
    # Basic scenario
    # 无头模式下场景无默认光源，添加额外光源(参照默认环境光设置)
    rep.create.light(light_type="Distant", temperature=6500, intensity=3000,
                     rotation=(315, 0, 0), scale=1, name="DefaultLight")

    WORKSHOP = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Industrial/Buildings/Warehouse/Warehouse01.usd'
    CONVEYORBELT_A23 = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Conveyors/ConveyorBelt_A/ConveyorBelt_A23_PR_NVD_01.usd'
    CONVEYORBELT_A37 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Conveyors/ConveyorBelt_A/ConveyorBelt_A37_PR_NVD_01.usd"
    CONVEYORBELT_A07 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Conveyors/ConveyorBelt_A/ConveyorBelt_A07_PR_NVD_01.usd"
    RACKLONG_A4 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Industrial/Racks/RackLong_A4.usd"
    RACKLONG_A2 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Industrial/Racks/RackLong_A2.usd"
    workshop = rep.create.from_usd(WORKSHOP)
    conveyor1, conveyor2 = get_num_same_node(item_num=2, usd=CONVEYORBELT_A23)
    conveyor3, conveyor4 = get_num_same_node(item_num=2, usd=CONVEYORBELT_A37)
    conveyor5, conveyor6 = get_num_same_node(item_num=2, usd=CONVEYORBELT_A07)
    racklong1, racklong2 = get_num_same_node(item_num=2, usd=RACKLONG_A4)
    racklong3, racklong4 = get_num_same_node(item_num=2, usd=RACKLONG_A2)

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
    with conveyor3:
        rep.modify.pose(
            position=(118, 0, -586),
            rotation=(-90, 0, 0)
        )
    with conveyor4:
        rep.modify.pose(
            position=(-198, 0, 685),
            rotation=(-90, 180, 0)
        )
    with conveyor5:
        rep.modify.pose(
            position=(-198, 0, 1085),
            rotation=(-90, 180, 0)
        )
    with conveyor6:
        rep.modify.pose(
            position=(118, 0, -985),
            rotation=(-90, 0, 0)
        )
    with racklong1:
        rep.modify.pose(
            position=(40, 0, -780),
            rotation=(-90, 0, 0)
        )
    with racklong2:
        rep.modify.pose(
            position=(300, 0, -780),
            rotation=(-90, -180, 0)
        )
    with racklong3:
        rep.modify.pose(
            position=(-380, 0, 880),
            rotation=(-90, -180, 0)
        )
    with racklong4:
        rep.modify.pose(
            position=(-110, 0, 880),
            rotation=(-90, 0, 0)
        )

    LowFullDesk = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Commercial/Storage/Contemporary/Contemporary_LowFullDesk.usd"
    desk1, desk2, desk3, desk4, desk5, desk6 = get_num_same_node(item_num=6, usd=LowFullDesk, semantics=[('name', 'desk')])
    with desk1:
        rep.modify.pose(
            position=(-100, 0, 135),
            rotation=(90, -180, -180)
        )
    with desk2:
        rep.modify.pose(
            position=(80, 0, 135),
            rotation=(90, -180, -180)
        )
    with desk3:
        rep.modify.pose(
            position=(260, 0, 135),
            rotation=(90, -180, -180)
        )
    with desk4:
        rep.modify.pose(
            position=(25, 0, -35),
            rotation=(-90, -180, 0)
        )
    with desk5:
        rep.modify.pose(
            position=(-155, 0, -35),
            rotation=(-90, -180, 0)
        )
    with desk6:
        rep.modify.pose(
            position=(-335, 0, -35),
            rotation=(-90, -180, 0)
        )

    def desk_randomize():
        prims = rep.get.prims(semantics=[('name', 'desk')])
        with prims:
            rep.modify.visibility(rep.distribution.choice([True, False]))
        return prims.node
    rep.randomizer.register(desk_randomize)

    def ladder_randomize(position: List[Tuple[float, float, float]], rotation: Tuple[float, float, float]):
        LADDER = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Ladders/GripStepRolling_A/GripStepRollingLadder_A03_PR_NVD_01.usd"
        ladder = rep.create.from_usd(usd=LADDER)
        with ladder:
            rep.modify.pose(
                position=rep.distribution.uniform(position[0], position[1]),
                rotation=rotation
            )
            rep.modify.visibility(rep.distribution.choice([True, False]))
        return ladder.node

    rep.randomizer.register(ladder_randomize)

    def anora_randomize(position: List[Tuple[float, float, float]], rotation: List[Tuple[float, float, float]],
                        count: int = 1):
        ANORA = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Commercial/Seating/Anora.usd"
        anora = rep.create.from_usd(usd=ANORA, count=count)
        with anora:
            rep.modify.pose(
                position=rep.distribution.uniform(position[0], position[1]),
                rotation=rep.distribution.uniform(rotation[0], rotation[1])
            )
            rep.modify.visibility(rep.distribution.choice([True, False]))
        return anora.node
    rep.randomizer.register(anora_randomize)

    # boxs - detection target
    box1, box2, box3, box4 = get_num_box_node(item_num=4, semantics=[('class', 'box'), ('area', 'box1')],
                                              filename_path=os.path.join(ROOTPATH, "Assets/BoxURLS/CardBox.txt"))

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

    box5, box6, box7, box8 = get_num_box_node(item_num=4, semantics=[('class', 'box'), ('area', 'box2')],
                                              filename_path=os.path.join(ROOTPATH, "Assets/BoxURLS/CardBox.txt"))
    with box5:
        rep.modify.pose(
            position=(-250, 78, 190),
            rotation=(0, -90, -90)
        )
    with box6:
        rep.modify.pose(
            position=(-250, 78, 310),
            rotation=(0, -90, -90)
        )

    with box7:
        rep.modify.pose(
            position=(168, 78, -90),
            rotation=(0, -90, -90)
        )
    with box8:
        rep.modify.pose(
            position=(168, 78, -210),
            rotation=(0, -90, -90)
        )

    def box_randomize1():
        shapes = rep.get.prims(semantics=[('area', 'box1')])
        with shapes:
            rep.modify.pose(
                position=rep.distribution.uniform((-5, -50, 0), (5, 50, 0)),
                rotation=rep.distribution.uniform((0, 0, -180), (0, 0, 180)),
                scale=rep.distribution.uniform(0.7, 1)
            )
            rep.modify.visibility(rep.distribution.choice([True, False]))
        return shapes.node
    rep.randomizer.register(box_randomize1)

    def box_randomize2():
        shapes = rep.get.prims(semantics=[('area', 'box2')])
        with shapes:
            rep.modify.pose(
                position=rep.distribution.uniform((-50, -5, 0), (50, 5, 0)),
                rotation=rep.distribution.uniform((0, 0, -180), (0, 0, 180)),
                scale=rep.distribution.uniform(0.7, 1)
            )
            rep.modify.visibility(rep.distribution.choice([True, False]))
        return shapes.node
    rep.randomizer.register(box_randomize2)

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

    # 创建相机并配置渲染
    camera = rep.create.camera(position=(-600, 200, -500), look_at=(-430, 78, -170))
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))

    with rep.trigger.on_frame(num_frames=NUM_FRAMES, rt_subframes=3):  # number of picture
        # basic scenario
        rep.randomizer.sphere_lights(4)  # number of lighting source
        rep.randomizer.anora_randomize([(-160, 0, 210), (300, 0, 315)], [(-90, -180, 0), (-90, 180, 0)], 3)
        rep.randomizer.anora_randomize([(-410, 0, -190), (60, 0, -110)], [(-90, -180, 0), (-90, 180, 0)], 3)
        rep.randomizer.ladder_randomize([(-515, 0, 730), (-515, 0, 1030)], (0, -90, -90))
        rep.randomizer.ladder_randomize([(30, 0, 730), (30, 0, 1030)], (180, 90, -90))
        rep.randomizer.ladder_randomize([(420, 0, -930), (420, 0, -630)], (-90, 90, 0))
        rep.randomizer.desk_randomize()

        # detection target
        rep.randomizer.box_randomize1()
        rep.randomizer.box_randomize2()

        # camera
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform((-600, 200, -500), (600, 500, 500)),
                look_at=rep.distribution.uniform((-430, 78, -170), (355, 78, 264)))

    # Initialize and attach writer for PASCAL VOC format data
    WriterRegistry.register(PascalWriter)
    writer = rep.WriterRegistry.get("PascalWriter")
    writer.initialize(
        output_dir=os.path.join(ROOTPATH, f"_output_{time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))}"),
        semantic_types=["class"],
        bbox_height_threshold=25,
        image_output_format="jpeg"
    )
    writer.attach([render_product])

    # required for headless mode
    rep.orchestrator.preview()
