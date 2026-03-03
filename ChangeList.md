## Summary of Changes: ORB-SLAM3 → SuperPoint-SLAM3 (TensorRT)

### New Files

1. **`include/SPextractor.h`** - TensorRT-based SuperPoint feature extractor header
2. **`src/SPextractor.cc`** - Full implementation: TensorRT inference, softmax heatmap, NMS, descriptor interpolation, L2 normalization, image pyramid, octree distribution
3. **`include/DUtils.h`** - Standalone random utility (replaces DBoW2/DUtils dependency)

### Modified Files

4. **`include/ORBVocabulary.h`** - `DBoW2::TemplatedVocabulary<FORB>` → `DBoW3::Vocabulary`
5. **`include/Frame.h`** - DBoW2 → DBoW3 types; `typedef SPextractor ORBextractor`
6. **`src/Frame.cc`** - `BFMatcher(NORM_HAMMING)` → `BFMatcher(NORM_L2)`; distance types `int` → `float`
7. **`include/Tracking.h`** - Added `mSPEnginePath`
8. **`src/Tracking.cc`** - Creates `SPextractor`; thresholds divided by 255 for SuperPoint probabilities
9. **`src/System.cc`** - Vocabulary loading: `loadFromTextFile()` → `load()` (DBoW3 format)
10. **`include/ORBmatcher.h`** - `DescriptorDistance` returns `float`; thresholds `const float`
11. **`src/ORBmatcher.cc`** - `TH_HIGH=0.7f, TH_LOW=0.3f`; L2 distance via `cv::norm`; all distance vars `float`
12. **`include/KeyFrame.h`** - DBoW2 → DBoW3 types
13. **`src/KeyFrameDatabase.cc`** - `DBoW2::` → `DBoW3::`
14. **`src/MapPoint.cc`** - Distance arrays `int` → `float`
15. **`include/SerializationUtils.h`** - Added boost serialization for `DBoW3::BowVector`/`DBoW3::FeatureVector`
16. **`src/Sim3Solver.cc`** - DUtils include updated to standalone `DUtils.h`
17. **`src/TwoViewReconstruction.cc`** - DUtils include updated to standalone `DUtils.h`
18. **`src/MLPnPsolver.cpp`** - Added `DUtils.h` include
19. **`CMakeLists.txt`** - CUDA/TensorRT; `SPextractor.cc`; links DBoW3, nvinfer, CUDA
20. **`build.sh`** - Builds DBoW3 instead of DBoW2

### Removed Files

21. **`include/ORBextractor.h`** - Replaced by SPextractor.h
22. **`src/ORBextractor.cc`** - Replaced by SPextractor.cc

### Removed Dependencies

- **DBoW2** - Replaced by DBoW3
- **ORB/BEBLID/TEBLID/AKAZE/LATCH descriptors** - Replaced by SuperPoint

### Architecture

| Aspect | ORB-SLAM3 (Before) | SuperPoint-SLAM3 (After) |
|--------|-------------------|--------------------------|
| **Feature extractor** | ORBextractor (FAST + ORB) | SPextractor (TensorRT SuperPoint) |
| **Descriptor** | 256-bit binary (`CV_8U`) | 256-dim float (`CV_32F`) |
| **Distance metric** | Hamming (int, 0-256) | L2 (float, 0-2.0) |
| **Thresholds** | TH_LOW=50, TH_HIGH=100 | TH_LOW=0.3, TH_HIGH=0.7 |
| **Vocabulary** | DBoW2 (text, binary desc) | DBoW3 (YAML/binary, float desc) |
| **Vocabulary file** | `ORBvoc.txt` | `superpoint_voc.yml.gz` |
