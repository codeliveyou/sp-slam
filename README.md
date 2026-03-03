# SuperPoint-SLAM3

A visual SLAM system based on [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3), replacing ORB features with **SuperPoint** (learned keypoints and descriptors) and **DBoW3** vocabulary for place recognition.

## Key Changes from ORB-SLAM3

| Component | ORB-SLAM3 | This Project |
|---|---|---|
| Feature extractor | ORB (handcrafted) | SuperPoint (deep learned, ONNX Runtime CPU) |
| Descriptor type | 256-bit binary | 256-dim float |
| Distance metric | Hamming | L2 |
| Vocabulary | DBoW2 (binary) | DBoW3 (float) |
| Vocabulary file | `ORBvoc.txt` | `superpoint_voc.yml.gz` |

The SLAM pipeline (tracking, local mapping, loop closing, IMU integration, multi-map) remains unchanged from ORB-SLAM3. Only the front-end feature extraction, descriptor matching, and vocabulary subsystems are replaced.

## Prerequisites

### Required

- **C++14** compiler (GCC 7+ or equivalent)
- **OpenCV 4.4+**
- **Eigen3** (>= 3.1.0)
- **Pangolin** -- visualization ([GitHub](https://github.com/stevenlovegrove/Pangolin))
- **Boost** -- serialization (`libboost-serialization-dev`)
- **ONNX Runtime** -- SuperPoint inference (CPU, no GPU required). Install via system package or download from [GitHub releases](https://github.com/microsoft/onnxruntime/releases)

### Optional

- **RealSense SDK 2.0** -- for Intel RealSense cameras
- **ROS** -- for ROS node examples

### Included in Thirdparty

- **DBoW3** -- bag-of-words vocabulary for float descriptors
- **g2o** -- graph optimization
- **Sophus** -- Lie group library

## Building

### 1. Prepare the SuperPoint ONNX model

You need a SuperPoint ONNX model file (`.onnx`). The expected I/O:

- **Input**: `[1, 1, H, W]` grayscale float image (0-1 normalized)
- **Output 0** (semi): `[1, 65, H/8, W/8]` keypoint heatmap
- **Output 1** (desc): `[1, 256, H/8, W/8]` descriptor map

Place the model file as `superpoint.onnx` in your working directory, or specify the path in your settings YAML via `SuperPoint.modelPath`.

No GPU or TensorRT is required -- the model runs on CPU via ONNX Runtime.

### 2. Build third-party libraries and the main project

```bash
chmod +x build.sh
./build.sh
```

This builds DBoW3, g2o, Sophus, and the main library + executables.

If ONNX Runtime is not in a standard system path, pass its location:

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=/path/to/onnxruntime
make -j4
```

The build produces `lib/libORB_SLAM3.so` and executables in `Examples/`.

## Running

Use `superpoint_voc.yml.gz` as the vocabulary file instead of the original ORB vocabulary.

### Monocular

```bash
./Examples/Monocular/mono_euroc superpoint_voc.yml.gz Examples/Monocular/EuRoC.yaml \
    /path/to/MH_01 Examples/Monocular/EuRoC_TimeStamps/MH01.txt
```

### Monocular-Inertial

```bash
./Examples/Monocular-Inertial/mono_inertial_euroc superpoint_voc.yml.gz \
    Examples/Monocular-Inertial/EuRoC.yaml \
    /path/to/MH_01 Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt
```

### Settings file

The feature extractor is configured through the same `ORBextractor.*` parameters in your YAML settings file:

```yaml
ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20      # mapped to SuperPoint threshold (divided by 255)
ORBextractor.minThFAST: 7       # fallback threshold (divided by 255)
```

The integer threshold values are converted to float probabilities internally: `threshold = value / 255.0`.

## Project Structure

```
├── include/
│   ├── SPextractor.h        # SuperPoint feature extractor (ONNX Runtime CPU)
│   ├── ORBmatcher.h         # Descriptor matching (L2 distance)
│   ├── ORBVocabulary.h      # typedef to DBoW3::Vocabulary
│   ├── Frame.h              # DBoW3 BowVector/FeatureVector
│   ├── KeyFrame.h           # DBoW3 BowVector/FeatureVector + serialization
│   ├── SerializationUtils.h # Boost serialization for DBoW3 types
│   ├── DUtils.h             # Standalone random utilities (replaces DBoW2/DUtils)
│   └── ...
├── src/
│   ├── SPextractor.cc       # ONNX Runtime inference + pyramid + octree distribution
│   ├── ORBmatcher.cc        # L2 distance, float thresholds
│   └── ...
├── Thirdparty/
│   ├── DBoW3/               # Float descriptor vocabulary
│   ├── g2o/                 # Graph optimization
│   └── Sophus/              # Lie groups
├── superpoint_voc.yml.gz    # Pre-built DBoW3 vocabulary for SuperPoint
└── build.sh
```

## Acknowledgments

This project is built on top of:

- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) by Campos et al.
- [SuperPoint](https://arxiv.org/abs/1712.07629) by DeTone, Malisiewicz, and Rabinovich (Magic Leap)
- [DBoW3](https://github.com/rmsalinas/DBow3) by Rafael Munoz-Salinas
- [SuperPoint-SLAM](https://github.com/KinglittleQ/SuperPoint_SLAM) reference implementation

### Citation

If you use this work in academic research, please cite:

```bibtex
@article{ORBSLAM3_TRO,
  title={{ORB-SLAM3}: An Accurate Open-Source Library for Visual, Visual-Inertial
         and Multi-Map {SLAM}},
  author={Campos, Carlos and Elvira, Richard and G{\'o}mez, Juan J. and Montiel,
          Jos{\'e} M. M. and Tard{\'o}s, Juan D.},
  journal={IEEE Transactions on Robotics},
  volume={37}, number={6}, pages={1874--1890}, year={2021}
}

@inproceedings{detone2018superpoint,
  title={SuperPoint: Self-Supervised Interest Point Detection and Description},
  author={DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={CVPR Workshops}, year={2018}
}
```

## License

ORB-SLAM3 is released under [GPLv3](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/LICENSE).
