# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmseg.registry import DATASETS
# from .custom import CustomDataset
from mmengine.dataset import BaseDataset, Compose
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class ORIEgoHOSDataset(BaseSegDataset):
    """EgoHOS dataset.

    Args:
        split (str): Split txt file for EgoHOS.
    """
    
    # CLASSES = ('background', 'aeroplane')

    # PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    #            [128, 0, 128]]

    METAINFO = dict(
        # classes = ('BG', 'Left_Object1', 'Right_Object1', 'Two_Object1'),   
        # palette=[[0, 0, 0], [255, 0, 255], [0, 255, 255], [0, 255, 0]],
        classes = ('background', 'Left_Hand', 'Right_Hand', \
               'Left_Object1', 'Right_Object1', 'Two_Object1', \
               'Left_Object2', 'Right_Object2', 'Two_Object2'),   
        palette=[[0, 0, 0], [255, 0, 0], [0, 0, 255], \
               [255, 0, 255], [0, 255, 255], [0, 255, 0], \
               [255, 204, 255], [204, 255, 255], [204, 255, 204]],
)

    # CLASSES = ('background', 'Left_Hand', 'Right_Hand', \
    #            'Left_Object1', 'Right_Object1', 'Two_Object1', \
    #            'Left_Object2', 'Right_Object2', 'Two_Object2')

    # PALETTE = [[0, 0, 0], [255, 0, 0], [0, 0, 255], \
    #            [255, 0, 255], [0, 255, 255], [0, 255, 0], \
    #            [255, 204, 255], [204, 255, 255], [204, 255, 204]]

    def __init__(self, **kwargs):
        super(ORIEgoHOSDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
        
    # def __init__(self,
    #              ann_file: str='',  # ann_file for obj
    #              img_suffix='.jpg',
    #              seg_map_suffix='.png',**kwargs) -> None:
