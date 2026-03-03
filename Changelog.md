# SuperPoint-SLAM3
Details of changes between the different versions.

### V2.0 - SuperPoint + DBoW3

Based on ORB-SLAM3 V1.0.

- Replaced ORB feature extraction with SuperPoint (TensorRT inference).
- Replaced DBoW2 (binary descriptors) with DBoW3 (float descriptors).
- Descriptor matching changed from Hamming distance to L2 distance.
- Removed all legacy descriptor backends (ORB, BEBLID, TEBLID, AKAZE, LATCH).
- Added CUDA and TensorRT as build dependencies.
- Added boost serialization support for DBoW3 BowVector/FeatureVector.
- Replaced DBoW2 DUtils dependency with standalone DUtils.h.

### V1.0, 22th December 2021 (ORB-SLAM3 upstream)

- OpenCV static matrices changed to Eigen matrices.
- New calibration file format.
- Added load/save map functionalities.
- Added examples of live SLAM using Intel Realsense cameras.
