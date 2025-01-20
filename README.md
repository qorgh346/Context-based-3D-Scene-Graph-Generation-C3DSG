C3DSG: 실내 환경 포인트 클라우드를 이용한 3차원 장면 그래프 생성 모델
(C3DSG: A 3D Scene Graph Generation Model Using Point Clouds of Indoor Environment)
(Journal of KIISE, 2023)

[모델 전체 구조도] 사진~

---

##Introduction

제안된 모델은 Point Transformer를 활용하여 3D 포인트 클라우드에서 기하학적 특징뿐만 아니라, 언어적 특징과 상대적 비교 특징과 같은 다양한 비-기하학적 특징을 함께 활용합니다. 또한, 물체 간의 공간적 맥락 정보를 효과적으로 추출하기 위해, NE-GAT 그래프 신경망을 이용하여 물체 노드와 간선 모두에 주의집중 메커니즘을 적용합니다.

##Dependencies

conda create -n csggn python=3.8
conda activate csggn
pip install -r requirement.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-geometric


##Prepare

###3RScan Dataset

1. Download 3Rscan and 3DSSG-Sub Annotation [Link](https://github.com/ShunChengWu/3DSSG)
	- Please make sure you agree the 3RScan Terms of Use first, and get the download script and put it right at the 3RScan main directory. 
	- python scripts/RUN_prepare_dataset_3RScan.py --download --thread 8

2. GT, Dense, Sparse Datset Download

	# For GT
# This script downloads preprocessed data for GT data generation, and generate GT data.
python scripts/RUN_prepare_GT_setup_3RScan.py --thread 16

# For Dense
# This script downloads the inseg.ply files and unzip them to your 3rscan folder, and 
generates training data.
python scripts/RUN_prepare_Dense_setup_3RScan.py -c configs/dataset/config_base_3RScan_inseg_l20.yaml --thread 16

# For Sparse
# This script downloads the 2dssg_orbslam3.[json,ply] files and unzip them to your 3rscan folder, and 
generates training data.
python scripts/RUN_prepare_Sparse_setup_3RScan.py -c configs/dataset/config_base_3RScan_orbslam_l20.yaml --thread 16

---
### Trained models
- Word Embedding : [Glove6B_50d]( 구글 드라이브 주소)
	-> 어디에 위치하는지 알려주기

- Geometric Feature Extraction : [PTv1](구글드라이브주소), [PTv2](구글드라이브주소)

- Ours Model Weight : [CSGGN](최종 모델 가중치 파일 담긴 구글 드라이브 주소)

##Run Code

# Train single
python main.py --mode train --config /path/to/your/config/file

# Eval one
python main.py --mode eval --config /path/to/your/config/file

---
## Paper

@article{백호준2023c3dsg,
  title={C3DSG: 실내 환경 포인트 클라우드를 이용한 3 차원 장면 그래프 생성 모델},
  author={백호준 and 김인철},
  journal={정보과학회논문지},
  volume={50},
  number={9},
  pages={758--770},
  year={2023}
}


## Acknowledgement

This repository is partly based on [3DSSG](~~) , [pointTransformer](~~) 