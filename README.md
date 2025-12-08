<div align="center">
  <picture>
    <!-- User prefers dark mode: -->
  <source srcset="https://raw.githubusercontent.com/vbti-development/onedl-mmdetection/main/docs/en/_static/image/onedl-mmdetection-banner-dark.png"  media="(prefers-color-scheme: dark)"/>

<img src="https://raw.githubusercontent.com/vbti-development/onedl-mmdetection/main/docs/en/_static/image/onedl-mmdetection-banner.png" alt="OneDL-Detection logo" height="200"/>
  </picture>

<div>&nbsp;</div>
  <div align="center">
    <a href="https://vbti.nl">
      <b><font size="5">VBTI Website</font></b>
    </a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://onedl.ai">
      <b><font size="5">OneDL platform</font></b>
    </a>
  </div>
<div>&nbsp;</div>

[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://onedl-mmdetection.readthedocs.io/en/latest/)
[![license](https://img.shields.io/github/license/VBTI-development/onedl-mmdetection.svg)](https://github.com/VBTI-development/onedl-mmdetection/blob/main/LICENSE)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/onedl-mmdetection)](https://pypi.org/project/onedl-mmdetection/)
[![PyPI](https://img.shields.io/pypi/v/onedl-mmdetection)](https://pypi.org/project/onedl-mmdetection)

[![Build Status](https://github.com/VBTI-development/onedl-mmdetection/actions/workflows/merge_stage_test.yml/badge.svg)](https://github.com/VBTI-development/onedl-mmdetection/actions/workflows/merge_stage_test.yml)
[![Docker Image](https://github.com/VBTI-development/onedl-mmdetection/actions/workflows/docker.yml/badge.svg)](https://hub.docker.com/r/vbti/onedl-mmdetection-cu129-torch280)
[![open issues](https://isitmaintained.com/badge/open/VBTI-development/onedl-mmdetection.svg)](https://github.com/VBTI-development/onedl-mmdetection/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/VBTI-development/onedl-mmdetection.svg)](https://github.com/VBTI-development/onedl-mmdetection/issues)

[📘 Documentation](https://onedl-mmdetection.readthedocs.io/en/latest/) |
[🛠️ Installation](https://onedl-mmdetection.readthedocs.io/en/latest/get_started.html) |
[👀 Model Zoo](https://onedl-mmdetection.readthedocs.io/en/latest/model_zoo.html) |
[🆕 Update News](https://onedl-mmdetection.readthedocs.io/en/latest/notes/changelog.html) |
[🚀Ongoing Projects](https://github.com/vbti-development/onedl-mmdetection/projects) |
[🤔 Reporting Issues](https://github.com/VBTI-development/onedl-mmdetection/issues/new/choose) |

[![Discord Logo](https://cdn.prod.website-files.com/6257adef93867e50d84d30e2/66e3d80db9971f10a9757c99_Symbol.svg)](https://discord.gg/8DvcVRs5Pm)

</div>

## Introduction

MMDetection is an open source object detection toolbox based on PyTorch.

The main branch works with **PyTorch 2.0+**.

<img src="https://user-images.githubusercontent.com/12907710/187674113-2074d658-f2fb-42d1-ac15-9c4a695e64d7.png"/>

<details open>
<summary>Major features</summary>

- **Modular Design**

  We decompose the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

- **Support of multiple tasks out of box**

  The toolbox directly supports multiple detection tasks such as **object detection**, **instance segmentation**, **panoptic segmentation**, and **semi-supervised object detection**.

- **High efficiency**

  All basic bbox and mask operations run on GPUs. The training speed is faster than or comparable to other codebases, including [Detectron2](https://github.com/facebookresearch/detectron2), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [SimpleDet](https://github.com/TuSimple/simpledet).

- **State of the art**

  The toolbox stems from the codebase developed by the *MMDet* team, who won [COCO Detection Challenge](http://cocodataset.org/#detection-leaderboard) in 2018, and we keep pushing it forward.
  The newly released [RTMDet](configs/rtmdet) also obtains new state-of-the-art results on real-time instance segmentation and rotated object detection tasks and the best parameter-accuracy trade-off on object detection.

</details>

Apart from MMDetection, we also released [MMEngine](https://github.com/vbti-development/onedl-mmengine) for model training and [MMCV](https://github.com/vbti-development/onedl-mmcv) for computer vision research, which are heavily depended on by this toolbox.

## What's New

The VBTI development team is reviving MMLabs code, making it work with
newer pytorch versions and fixing bugs. We are only a small team, so your help
is appreciated.

💎 **We have released the pre-trained weights for MM-Grounding-DINO Swin-B and Swin-L, welcome to try and give feedback.**

### Highlight

**v3.3.0** was released in 5/1/2024:

**[MM-Grounding-DINO: An Open and Comprehensive Pipeline for Unified Object Grounding and Detection](https://arxiv.org/abs/2401.02361)**

Grounding DINO is a grounding pre-training model that unifies 2d open vocabulary object detection and phrase grounding, with wide applications. However, its training part has not been open sourced. Therefore, we propose MM-Grounding-DINO, which not only serves as an open source replication version of Grounding DINO, but also achieves significant performance improvement based on reconstructed data types, exploring different dataset combinations and initialization strategies. Moreover, we conduct evaluations from multiple dimensions, including OOD, REC, Phrase Grounding, OVD, and Fine-tune, to fully excavate the advantages and disadvantages of Grounding pre-training, hoping to provide inspiration for future work.

code: [mm_grounding_dino/README.md](configs/mm_grounding_dino/README.md)

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/fb14d1ee-5469-44d2-b865-aac9850c429c"/>
</div>

We are excited to announce our latest work on real-time object recognition tasks, **RTMDet**, a family of fully convolutional single-stage detectors. RTMDet not only achieves the best parameter-accuracy trade-off on object detection from tiny to extra-large model sizes but also obtains new state-of-the-art performance on instance segmentation and rotated object detection tasks. Details can be found in the [technical report](https://arxiv.org/abs/2212.07784). Pre-trained models are [here](configs/rtmdet).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/real-time-instance-segmentation-on-mscoco)](https://paperswithcode.com/sota/real-time-instance-segmentation-on-mscoco?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-dota-1)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dota-1?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-hrsc2016)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-hrsc2016?p=rtmdet-an-empirical-study-of-designing-real)

| Task                     | Dataset | AP                                   | FPS(TRT FP16 BS1 3090) |
| ------------------------ | ------- | ------------------------------------ | ---------------------- |
| Object Detection         | COCO    | 52.8                                 | 322                    |
| Instance Segmentation    | COCO    | 44.6                                 | 188                    |
| Rotated Object Detection | DOTA    | 78.9(single-scale)/81.3(multi-scale) | 121                    |

<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/208044554-1e8de6b5-48d8-44e4-a7b5-75076c7ebb71.png"/>
</div>

## Installation

Please refer to [Installation](https://onedl-mmdetection.readthedocs.io/en/latest/get_started.html) for installation instructions.

## Getting Started

Please see [Overview](https://onedl-mmdetection.readthedocs.io/en/latest/get_started.html) for the general introduction of MMDetection.

For detailed user guides and advanced guides, please refer to our [documentation](https://onedl-mmdetection.readthedocs.io/en/latest/):

- User Guides

  <details>

  - [Train & Test](https://onedl-mmdetection.readthedocs.io/en/latest/user_guides/index.html#train-test)
    - [Learn about Configs](https://onedl-mmdetection.readthedocs.io/en/latest/user_guides/config.html)
    - [Inference with existing models](https://onedl-mmdetection.readthedocs.io/en/latest/user_guides/inference.html)
    - [Dataset Prepare](https://onedl-mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html)
    - [Test existing models on standard datasets](https://onedl-mmdetection.readthedocs.io/en/latest/user_guides/test.html)
    - [Train predefined models on standard datasets](https://onedl-mmdetection.readthedocs.io/en/latest/user_guides/train.html)
    - [Train with customized datasets](https://onedl-mmdetection.readthedocs.io/en/latest/user_guides/train.html#train-with-customized-datasets)
    - [Train with customized models and standard datasets](https://onedl-mmdetection.readthedocs.io/en/latest/user_guides/new_model.html)
    - [Finetuning Models](https://onedl-mmdetection.readthedocs.io/en/latest/user_guides/finetune.html)
    - [Test Results Submission](https://onedl-mmdetection.readthedocs.io/en/latest/user_guides/test_results_submission.html)
    - [Weight initialization](https://onedl-mmdetection.readthedocs.io/en/latest/user_guides/init_cfg.html)
    - [Use a single stage detector as RPN](https://onedl-mmdetection.readthedocs.io/en/latest/user_guides/single_stage_as_rpn.html)
    - [Semi-supervised Object Detection](https://onedl-mmdetection.readthedocs.io/en/latest/user_guides/semi_det.html)
  - [Useful Tools](https://onedl-mmdetection.readthedocs.io/en/latest/user_guides/index.html#useful-tools)

  </details>

- Advanced Guides

  <details>

  - [Basic Concepts](https://onedl-mmdetection.readthedocs.io/en/latest/advanced_guides/index.html#basic-concepts)
  - [Component Customization](https://onedl-mmdetection.readthedocs.io/en/latest/advanced_guides/index.html#component-customization)
  - [How to](https://onedl-mmdetection.readthedocs.io/en/latest/advanced_guides/index.html#how-to)

  </details>

We also provide object detection colab tutorial [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](demo/MMDet_Tutorial.ipynb) and instance segmentation colab tutorial [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](demo/MMDet_InstanceSeg_Tutorial.ipynb).

To migrate from MMDetection 2.x, please refer to [migration](https://onedl-mmdetection.readthedocs.io/en/latest/migration.html).

## Overview of Benchmark and Model Zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

<div align="center">
  <b>Architectures</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Object Detection</b>
      </td>
      <td>
        <b>Instance Segmentation</b>
      </td>
      <td>
        <b>Panoptic Segmentation</b>
      </td>
      <td>
        <b>Other</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/fast_rcnn">Fast R-CNN (ICCV'2015)</a></li>
            <li><a href="configs/faster_rcnn">Faster R-CNN (NeurIPS'2015)</a></li>
            <li><a href="configs/rpn">RPN (NeurIPS'2015)</a></li>
            <li><a href="configs/ssd">SSD (ECCV'2016)</a></li>
            <li><a href="configs/retinanet">RetinaNet (ICCV'2017)</a></li>
            <li><a href="configs/cascade_rcnn">Cascade R-CNN (CVPR'2018)</a></li>
            <li><a href="configs/yolo">YOLOv3 (ArXiv'2018)</a></li>
            <li><a href="configs/cornernet">CornerNet (ECCV'2018)</a></li>
            <li><a href="configs/grid_rcnn">Grid R-CNN (CVPR'2019)</a></li>
            <li><a href="configs/guided_anchoring">Guided Anchoring (CVPR'2019)</a></li>
            <li><a href="configs/fsaf">FSAF (CVPR'2019)</a></li>
            <li><a href="configs/centernet">CenterNet (CVPR'2019)</a></li>
            <li><a href="configs/libra_rcnn">Libra R-CNN (CVPR'2019)</a></li>
            <li><a href="configs/tridentnet">TridentNet (ICCV'2019)</a></li>
            <li><a href="configs/fcos">FCOS (ICCV'2019)</a></li>
            <li><a href="configs/reppoints">RepPoints (ICCV'2019)</a></li>
            <li><a href="configs/free_anchor">FreeAnchor (NeurIPS'2019)</a></li>
            <li><a href="configs/cascade_rpn">CascadeRPN (NeurIPS'2019)</a></li>
            <li><a href="configs/foveabox">Foveabox (TIP'2020)</a></li>
            <li><a href="configs/double_heads">Double-Head R-CNN (CVPR'2020)</a></li>
            <li><a href="configs/atss">ATSS (CVPR'2020)</a></li>
            <li><a href="configs/nas_fcos">NAS-FCOS (CVPR'2020)</a></li>
            <li><a href="configs/centripetalnet">CentripetalNet (CVPR'2020)</a></li>
            <li><a href="configs/autoassign">AutoAssign (ArXiv'2020)</a></li>
            <li><a href="configs/sabl">Side-Aware Boundary Localization (ECCV'2020)</a></li>
            <li><a href="configs/dynamic_rcnn">Dynamic R-CNN (ECCV'2020)</a></li>
            <li><a href="configs/detr">DETR (ECCV'2020)</a></li>
            <li><a href="configs/paa">PAA (ECCV'2020)</a></li>
            <li><a href="configs/vfnet">VarifocalNet (CVPR'2021)</a></li>
            <li><a href="configs/sparse_rcnn">Sparse R-CNN (CVPR'2021)</a></li>
            <li><a href="configs/yolof">YOLOF (CVPR'2021)</a></li>
            <li><a href="configs/yolox">YOLOX (CVPR'2021)</a></li>
            <li><a href="configs/deformable_detr">Deformable DETR (ICLR'2021)</a></li>
            <li><a href="configs/tood">TOOD (ICCV'2021)</a></li>
            <li><a href="configs/ddod">DDOD (ACM MM'2021)</a></li>
            <li><a href="configs/rtmdet">RTMDet (ArXiv'2022)</a></li>
            <li><a href="configs/conditional_detr">Conditional DETR (ICCV'2021)</a></li>
            <li><a href="configs/dab_detr">DAB-DETR (ICLR'2022)</a></li>
            <li><a href="configs/dino">DINO (ICLR'2023)</a></li>
            <li><a href="configs/glip">GLIP (CVPR'2022)</a></li>
            <li><a href="configs/ddq">DDQ (CVPR'2023)</a></li>
            <li><a href="projects/DiffusionDet">DiffusionDet (ArXiv'2023)</a></li>
            <li><a href="projects/EfficientDet">EfficientDet (CVPR'2020)</a></li>
            <li><a href="projects/ViTDet">ViTDet (ECCV'2022)</a></li>
            <li><a href="projects/Detic">Detic (ECCV'2022)</a></li>
            <li><a href="projects/CO-DETR">CO-DETR (ICCV'2023)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/mask_rcnn">Mask R-CNN (ICCV'2017)</a></li>
          <li><a href="configs/cascade_rcnn">Cascade Mask R-CNN (CVPR'2018)</a></li>
          <li><a href="configs/ms_rcnn">Mask Scoring R-CNN (CVPR'2019)</a></li>
          <li><a href="configs/htc">Hybrid Task Cascade (CVPR'2019)</a></li>
          <li><a href="configs/yolact">YOLACT (ICCV'2019)</a></li>
          <li><a href="configs/instaboost">InstaBoost (ICCV'2019)</a></li>
          <li><a href="configs/solo">SOLO (ECCV'2020)</a></li>
          <li><a href="configs/point_rend">PointRend (CVPR'2020)</a></li>
          <li><a href="configs/detectors">DetectoRS (ArXiv'2020)</a></li>
          <li><a href="configs/solov2">SOLOv2 (NeurIPS'2020)</a></li>
          <li><a href="configs/scnet">SCNet (AAAI'2021)</a></li>
          <li><a href="configs/queryinst">QueryInst (ICCV'2021)</a></li>
          <li><a href="configs/mask2former">Mask2Former (ArXiv'2021)</a></li>
          <li><a href="configs/condinst">CondInst (ECCV'2020)</a></li>
          <li><a href="projects/SparseInst">SparseInst (CVPR'2022)</a></li>
          <li><a href="configs/rtmdet">RTMDet (ArXiv'2022)</a></li>
          <li><a href="configs/boxinst">BoxInst (CVPR'2021)</a></li>
          <li><a href="projects/ConvNeXt-V2">ConvNeXt-V2 (Arxiv'2023)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/panoptic_fpn">Panoptic FPN (CVPR'2019)</a></li>
          <li><a href="configs/maskformer">MaskFormer (NeurIPS'2021)</a></li>
          <li><a href="configs/mask2former">Mask2Former (ArXiv'2021)</a></li>
          <li><a href="configs/XDecoder">XDecoder (CVPR'2023)</a></li>
        </ul>
      </td>
      <td>
        </ul>
          <li><b>Contrastive Learning</b></li>
        <ul>
        <ul>
          <li><a href="configs/selfsup_pretrain">SwAV (NeurIPS'2020)</a></li>
          <li><a href="configs/selfsup_pretrain">MoCo (CVPR'2020)</a></li>
          <li><a href="configs/selfsup_pretrain">MoCov2 (ArXiv'2020)</a></li>
        </ul>
        </ul>
        </ul>
          <li><b>Distillation</b></li>
        <ul>
        <ul>
          <li><a href="configs/ld">Localization Distillation (CVPR'2022)</a></li>
          <li><a href="configs/lad">Label Assignment Distillation (WACV'2022)</a></li>
        </ul>
        </ul>
          <li><b>Semi-Supervised Object Detection</b></li>
        <ul>
        <ul>
          <li><a href="configs/soft_teacher">Soft Teacher (ICCV'2021)</a></li>
        </ul>
        </ul>
          <li><b>Lane detection</b></li>
        <ul>
        <ul>
          <li><a href="configs/clrnet">CLRNet (CVPR'2022)</a></li>
        </ul>
        </ul>
      </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

<div align="center">
  <b>Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Necks</b>
      </td>
      <td>
        <b>Loss</b>
      </td>
      <td>
        <b>Common</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li>VGG (ICLR'2015)</li>
        <li>ResNet (CVPR'2016)</li>
        <li>ResNeXt (CVPR'2017)</li>
        <li>MobileNetV2 (CVPR'2018)</li>
        <li><a href="configs/hrnet">HRNet (CVPR'2019)</a></li>
        <li><a href="configs/empirical_attention">Generalized Attention (ICCV'2019)</a></li>
        <li><a href="configs/gcnet">GCNet (ICCVW'2019)</a></li>
        <li><a href="configs/res2net">Res2Net (TPAMI'2020)</a></li>
        <li><a href="configs/regnet">RegNet (CVPR'2020)</a></li>
        <li><a href="configs/resnest">ResNeSt (ArXiv'2020)</a></li>
        <li><a href="configs/pvt">PVT (ICCV'2021)</a></li>
        <li><a href="configs/swin">Swin (CVPR'2021)</a></li>
        <li><a href="configs/pvt">PVTv2 (ArXiv'2021)</a></li>
        <li><a href="configs/resnet_strikes_back">ResNet strikes back (ArXiv'2021)</a></li>
        <li><a href="configs/efficientnet">EfficientNet (ArXiv'2021)</a></li>
        <li><a href="configs/convnext">ConvNeXt (CVPR'2022)</a></li>
        <li><a href="projects/ConvNeXt-V2">ConvNeXtv2 (ArXiv'2023)</a></li>
      </ul>
      </td>
      <td>
      <ul>
        <li><a href="configs/pafpn">PAFPN (CVPR'2018)</a></li>
        <li><a href="configs/nas_fpn">NAS-FPN (CVPR'2019)</a></li>
        <li><a href="configs/carafe">CARAFE (ICCV'2019)</a></li>
        <li><a href="configs/fpg">FPG (ArXiv'2020)</a></li>
        <li><a href="configs/groie">GRoIE (ICPR'2020)</a></li>
        <li><a href="configs/dyhead">DyHead (CVPR'2021)</a></li>
        <li><a href="configs/clrnet">CLRHead (CVPR'2022)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/ghm">GHM (AAAI'2019)</a></li>
          <li><a href="configs/gfl">Generalized Focal Loss (NeurIPS'2020)</a></li>
          <li><a href="configs/seesaw_loss">Seasaw Loss (CVPR'2021)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/faster_rcnn/faster-rcnn_r50_fpn_ohem_1x_coco.py">OHEM (CVPR'2016)</a></li>
          <li><a href="configs/gn">Group Normalization (ECCV'2018)</a></li>
          <li><a href="configs/dcn">DCN (ICCV'2017)</a></li>
          <li><a href="configs/dcnv2">DCNv2 (CVPR'2019)</a></li>
          <li><a href="configs/gn+ws">Weight Standardization (ArXiv'2019)</a></li>
          <li><a href="configs/pisa">Prime Sample Attention (CVPR'2020)</a></li>
          <li><a href="configs/strong_baselines">Strong Baselines (CVPR'2021)</a></li>
          <li><a href="configs/resnet_strikes_back">Resnet strikes back (ArXiv'2021)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

Some other methods are also supported in [projects using MMDetection](./docs/en/notes/projects.md).

## FAQ

Please refer to [FAQ](docs/en/notes/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMDetection. Ongoing projects can be found in out [GitHub Projects](https://github.com/vbti-development/onedl-mmdetection/projects). Welcome community users to participate in these projects. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{OneDL-MMDetection}: OneDL MMLab Detection Toolbox and Benchmark},
  author  = {OneDL MMDetection Contributors},
  year={2025}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in VBTI-development

- [OneDL-MMEngine](https://github.com/vbti-development/onedl-mmengine): Foundational library for training deep learning models.
- [OneDL-MMCV](https://github.com/vbti-development/onedl-mmcv): Foundational library for computer vision.
- [OneDL-MMPreTrain](https://github.com/vbti-development/onedl-mmpretrain): Pre-training toolbox and benchmark.
- [OneDL-MMDetection](https://github.com/vbti-development/onedl-mmdetection): Detection toolbox and benchmark.
- [OneDL-MMRotate](https://github.com/vbti-development/onedl-mmrotate): Rotated object detection toolbox and benchmark.
- [OneDL-MMSegmentation](https://github.com/vbti-development/onedl-mmsegmentation): Semantic segmentation toolbox and benchmark.
- [OneDL-MMDeploy](https://github.com/vbti-development/onedl-mmdeploy): Model deployment framework.
- [OneDL-MIM](https://github.com/vbti-development/onedl-mim): MIM installs VBTI packages.
