ROOT_PATH = '/home/baebro/hojun_ws/3DSSG/'
DATA_PATH = '/home/baebro/3RScan/data/new_3RScan'
SCANNET_DATA_PATH = '/path/to/scannet' 
SCANNET_SPLIT_TRAIN = '/path/to/scannet/Tasks/Benchmarkscannetv2_train.txt'
SCANNET_SPLIT_VAL = '/path/to/scannet/Tasks/Benchmark/scannetv2_val.txt'

FILE_PATH = ROOT_PATH+'files/'
Scan3RJson_PATH = FILE_PATH+'3RScan.json'
LABEL_MAPPING_FILE = FILE_PATH+'3RScan.v2 Semantic Classes - Mapping.csv'
CLASS160_FILE = FILE_PATH+'classes160.txt'

# 3RScan file names
LABEL_FILE_NAME_RAW = 'labels.instances.annotated.v2.ply'
LABEL_FILE_NAME = 'labels.instances.align.annotated.v2.ply'
SEMSEG_FILE_NAME = 'semseg.v2.json'
MTL_NAME = 'mesh.refined.mtl'
OBJ_NAME = 'mesh.refined.obj'
TEXTURE_NAME = 'mesh.refined_0.png'

# ScanNet file names
SCANNET_SEG_SUBFIX = '_vh_clean_2.0.010000.segs.json'
SCANNET_AGGRE_SUBFIX = '.aggregation.json'
SCANNET_PLY_SUBFIX = '_vh_clean_2.labels.ply'


NAME_SAME_PART = 'same part'

from utils.pointcloud_transform import *

train_transform=[
    CenterShift(apply_z=True),
    RandomScale(scale=[0.9,1.1]),
    RandomFlip(p=0.5),
    RandomJitter(sigma=0.005, clip=0.02),
    ChromaticAutoContrast(p=0.2,blend_factor=None),
    ChromaticTranslation(p=0.95, ratio=0.05),
    ChromaticJitter(p=0.95,std=0.05),
    Voxelize(voxel_size=0.04,hash_type='fnv',mode='train',keys=('coord', 'color'),return_discrete_coord=True),
    SphereCrop(point_max=100000, mode='random'),
    CenterShift(apply_z=False),
    NormalizeColor(),
    ToTensor(),
    Collect(keys=('coord', 'discrete_coord'),feat_keys=['coord', 'color'])
    ]


test_transform = [
    #CenterShift(apply_z=True),
    #Voxelize(voxel_size=0.04,hash_type='fnv',mode='train',keys=('coord', 'color', 'label'),return_discrete_coord=True),
    # NormalizeColor(),
    ToTensor(),
    Collect(keys=('coord'), offset_keys_dict=dict(offset='coord'), feat_keys=['coord', 'color'])
]

#     Collect(keys=('coord', 'discrete_coord', 'label'), offset_keys_dict=dict(offset='coord'), feat_keys=['coord', 'color'])