# Copyright (c) OpenMMLab. All rights reserved.
from .det_data_sample import DetDataSample, OptSampleList, SampleList
from .reid_data_sample import ReIDDataSample
from .track_data_sample import (OptTrackSampleList, TrackDataSample,
                                TrackSampleList)

# A custom value to distinguish instance ID and category ID in panoptic
# segmentation maps; needs to be greater than the number of categories.
# For a pixel in the panoptic result map:
#   pan_id = ins_id * INSTANCE_OFFSET + cat_id
INSTANCE_OFFSET = 1000

__all__ = [
    'DetDataSample', 'SampleList', 'OptSampleList', 'TrackDataSample',
    'TrackSampleList', 'OptTrackSampleList', 'ReIDDataSample',
    'INSTANCE_OFFSET'
]
