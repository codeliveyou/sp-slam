#ifndef SPEXTRACTOR_H
#define SPEXTRACTOR_H

#include <vector>
#include <list>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
using namespace std;
#ifdef USE_CUDA
#include <onnxruntime_c_api.h>
#endif

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
                const std::string& modelPath = "superpoint.onnx");

    ~SPextractor(){}

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

    std::string mModelPath;
    std::unique_ptr<Ort::Env> mEnv;
    std::unique_ptr<Ort::Session> mSession;
    Ort::SessionOptions mSessionOptions;

    int mInputH = 0;
    int mInputW = 0;
    bool mDynamicShape = false;
    bool mUseCUDA = false;
    int mDeviceId = 0;

    std::string mInputName;
    std::string mSemiName;
    std::string mDescName;

    std::vector<const char*> mInputNames;
    std::vector<const char*> mOutputNames;
    std::vector<std::string> mInputNamesStr;
    std::vector<std::string> mOutputNamesStr;

    void loadModel(const std::string& modelPath);
    void runInference(const cv::Mat& img, cv::Mat& heatmap, cv::Mat& desc);
};

} // namespace ORB_SLAM3

#endif
