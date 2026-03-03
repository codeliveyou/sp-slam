#ifndef SPEXTRACTOR_H
#define SPEXTRACTOR_H

#include <vector>
#include <list>
#include <string>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>

namespace ORB_SLAM3
{

class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class SPextractor
{
public:
    SPextractor(int nfeatures, float scaleFactor, int nlevels,
                float iniThFAST, float minThFAST,
                const std::string& enginePath = "superpoint.engine");

    ~SPextractor();

    int operator()(cv::InputArray image, cv::InputArray mask,
                   std::vector<cv::KeyPoint>& keypoints,
                   cv::OutputArray descriptors, std::vector<int> &vLappingArea);

    int inline GetLevels(){ return nlevels; }

    float inline GetScaleFactor(){ return scaleFactor; }

    std::vector<float> inline GetScaleFactors(){ return mvScaleFactor; }
    std::vector<float> inline GetInverseScaleFactors(){ return mvInvScaleFactor; }
    std::vector<float> inline GetScaleSigmaSquares(){ return mvLevelSigma2; }
    std::vector<float> inline GetInverseScaleSigmaSquares(){ return mvInvLevelSigma2; }

    std::vector<cv::Mat> mvImagePyramid;

protected:
    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>>& allKeypoints, cv::Mat &_desc);
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys,
                                                const int &minX, const int &maxX,
                                                const int &minY, const int &maxY,
                                                const int &nFeatures, const int &level);

    void detect(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints,
                cv::Mat& descriptors, float threshold);

    int nfeatures;
    double scaleFactor;
    int nlevels;
    float iniThFAST;
    float minThFAST;

    std::vector<int> mnFeaturesPerLevel;
    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    std::string mEnginePath;

    // TRT objects: declared in dependency order so destruction is safe
    // (class members are destroyed in reverse declaration order)
    nvinfer1::IRuntime* mRuntime = nullptr;
    nvinfer1::ICudaEngine* mEngine = nullptr;
    nvinfer1::IExecutionContext* mContext = nullptr;

    cudaStream_t mCudaStream = nullptr;

    int mInputH = 0;
    int mInputW = 0;

    // Pre-allocated GPU buffers (reused across frames)
    void* mDevInput = nullptr;
    void* mDevHeatmap = nullptr;
    void* mDevDesc = nullptr;
    size_t mBufInputSize = 0;
    size_t mBufHeatmapSize = 0;
    size_t mBufDescSize = 0;

    void loadEngine(const std::string& enginePath);
    void runInference(const cv::Mat& img, cv::Mat& heatmap, cv::Mat& desc);
    void allocateBuffers(int H, int W);
    void freeBuffers();
};

} // namespace ORB_SLAM3

#endif
