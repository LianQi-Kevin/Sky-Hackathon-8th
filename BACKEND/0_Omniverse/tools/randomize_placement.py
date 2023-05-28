"""
randomize_placement 方法用来从指定的 usd 文件夹内读取所有 usd 文件并批量随即摆放

* 由于暂无法为为所有随机对象配置 semantics, 故暂时舍弃该方法

Examples:
    >>> import omni.replicator.core as rep
    >>> rep.randomizer.register(randomize_placement)
    >>> with rep.trigger.on_frame(num_frames=50):
    ...     rep.randomizer.randomize_placement(props=os.path.join(os.getcwd(), "Assets/Cartons"),
    ...                                        position=[(-430, 78, 57), (355, 78, 57)],
    ...                                        rotation=[(90, -180, -180), (90, 180, -180)],
    ...                                        size=5, semantics=[('class', 'box')])

https://docs.omniverse.nvidia.com/app_code/prod_extensions/ext_replicator/shrubs_and_worker_example.html#randomizing-appearance-placement-and-orientation-of-an-existing-3d-assets-with-a-built-in-writer
"""

import os
import glob
from typing import List, Tuple

import omni.replicator.core as rep
from omni.replicator.core.scripts.create import ReplicatorItem


def _multi_suffixes(path: str, suffixes: list) -> list:
    """
    获取多种文件后缀名的文件路径列表
    """
    return [file for suffix in suffixes for file in glob.glob(os.path.join(path, suffix))]


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
        # rep.modify.semantics(semantics)
    return instances.node
