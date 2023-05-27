import glob
import os
from typing import List, Tuple
import omni.replicator.core as rep
from omni.replicator.core.scripts.create import ReplicatorItem

# setup random view range for camera: low point, high point
sequential_pos = [(-800, 220, -271), (800, 220, 500)]

# position of look-at target
look_at_position = (-212, 78, 57)


def _multi_suffixes(path: str, suffixes: list) -> list:
    """
    获取多种文件后缀名的文件路径列表
    """
    return [file for suffix in suffixes for file in glob.glob(os.path.join(path, suffix))]


with rep.new_layer():
    # headless mode light
    rep.create.light(light_type="Distant", temperature=6500, intensity=3000,
                     rotation=(315, 0, 0), scale=1)

    # basic scenario
    CONVEYOR = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Conveyors/ConveyorBelt_A/ConveyorBelt_A23_PR_NVD_01.usd'
    conveyor1 = rep.create.from_usd(CONVEYOR)
    conveyor2 = rep.create.from_usd(CONVEYOR)
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

    # randomize box placement
    def randomize_placement(props: str, position: List[Tuple[float, float, float]],
                            rotation: List[Tuple[float, float, float]], semantics: List[Tuple[str, str]] = None,
                            size: int = 1, mode='point_instance') -> ReplicatorItem:
        """
        :param props: The folder path of USDs, [*.usd, *.usdc, *.usda]
        :param position: The samples distribution position, [lower, upper]
        :param rotation: The samples distribution rotation, [lower, upper]
        :param size: The number of samples to sample
        :param semantics: The samples semantics
        :param mode: The instantiation mode. Choose from [scene_instance, point_instance, reference]
        """
        usd_file_list = _multi_suffixes(path=props, suffixes=["*.usd", "*.usdc", "*.usda"])
        assert len(usd_file_list) != 0, f"usd file nor found in {props}"
        instances = rep.randomizer.instantiate(paths=rep.utils.get_usd_files(props, recursive=True), size=size, mode=mode)
        with instances:
            rep.modify.pose(
                position=rep.distribution.uniform(position[0], position[1]),
                rotation=rep.distribution.uniform(rotation[0], rotation[1])
            )
            rep.modify.semantics(semantics)
        return instances.node
    rep.randomizer.register(randomize_placement)

    # set camera
    camera = rep.create.camera(position=sequential_pos[0], look_at=look_at_position)
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))

    # set randomize action
    with rep.trigger.on_frame(50):
        rep.randomizer.randomize_placement(props=os.path.join(os.getcwd(), "Assets/Cartons"),
                                           position=[(-430, 78, 57), (355, 78, 57)],
                                           rotation=[(90, -180, -180), (90, 180, -180)],
                                           size=5, semantics=[('class', 'box')])
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform(sequential_pos[0], sequential_pos[1]),
                look_at=look_at_position)

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
