#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "SPextractor.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM3
{

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;
const int SP_DESC_DIM = 256;

void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    for(size_t i=0;i<vKeys.size();i++)
    {
        const cv::KeyPoint &kp = vKeys[i];
        if(kp.pt.x<n1.UR.x)
        {
            if(kp.pt.y<n1.BL.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if(kp.pt.y<n1.BL.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    if(n1.vKeys.size()==1) n1.bNoMore = true;
    if(n2.vKeys.size()==1) n2.bNoMore = true;
    if(n3.vKeys.size()==1) n3.bNoMore = true;
    if(n4.vKeys.size()==1) n4.bNoMore = true;
}

class TRTLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if(severity <= Severity::kWARNING)
            std::cerr << "[TensorRT] " << msg << std::endl;
    }
};

static TRTLogger gLogger;

SPextractor::SPextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                         float _iniThFAST, float _minThFAST,
                         const std::string& enginePath)
    : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
      iniThFAST(_iniThFAST), minThFAST(_minThFAST), mEnginePath(enginePath)
{
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    cudaStreamCreate(&mCudaStream);
    loadEngine(mEnginePath);
}

SPextractor::~SPextractor()
{
    freeBuffers();

    if(mCudaStream) {
        cudaStreamDestroy(mCudaStream);
        mCudaStream = nullptr;
    }

    // Destroy in dependency order: context -> engine -> runtime
    if(mContext) { delete mContext; mContext = nullptr; }
    if(mEngine)  { delete mEngine;  mEngine  = nullptr; }
    if(mRuntime) { delete mRuntime; mRuntime = nullptr; }
}

void SPextractor::loadEngine(const std::string& enginePath)
{
    std::ifstream file(enginePath, std::ios::binary);
    if(!file.good())
    {
        std::cerr << "SPextractor: Cannot open TensorRT engine file: " << enginePath << std::endl;
        return;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    mRuntime = nvinfer1::createInferRuntime(gLogger);
    if(!mRuntime)
    {
        std::cerr << "SPextractor: Failed to create TensorRT runtime" << std::endl;
        return;
    }

    mEngine = mRuntime->deserializeCudaEngine(engineData.data(), size);
    if(!mEngine)
    {
        std::cerr << "SPextractor: Failed to deserialize TensorRT engine." << std::endl;
        std::cerr << "  Make sure the engine was built with TensorRT 10.x and matches your GPU." << std::endl;
        return;
    }

    mContext = mEngine->createExecutionContext();
    if(!mContext)
    {
        std::cerr << "SPextractor: Failed to create TensorRT execution context" << std::endl;
        return;
    }

    const char* inputName = mEngine->getIOTensorName(0);
    auto inputDims = mEngine->getTensorShape(inputName);
    if(inputDims.nbDims == 4) {
        mInputH = static_cast<int>(inputDims.d[2]);
        mInputW = static_cast<int>(inputDims.d[3]);
    }

    std::cout << "SPextractor: TensorRT engine loaded (" << mEngine->getNbIOTensors()
              << " IO tensors). Input: " << mInputH << "x" << mInputW << std::endl;
}

void SPextractor::allocateBuffers(int H, int W)
{
    int Hc = H / 8;
    int Wc = W / 8;

    size_t inputSize   = 1 * 1 * H * W * sizeof(float);
    size_t heatmapSize = 1 * 65 * Hc * Wc * sizeof(float);
    size_t descSize    = 1 * 256 * Hc * Wc * sizeof(float);

    if(inputSize != mBufInputSize || heatmapSize != mBufHeatmapSize || descSize != mBufDescSize)
    {
        freeBuffers();

        cudaMalloc(&mDevInput, inputSize);
        cudaMalloc(&mDevHeatmap, heatmapSize);
        cudaMalloc(&mDevDesc, descSize);

        mBufInputSize   = inputSize;
        mBufHeatmapSize = heatmapSize;
        mBufDescSize    = descSize;
    }
}

void SPextractor::freeBuffers()
{
    if(mDevInput)   { cudaFree(mDevInput);   mDevInput   = nullptr; }
    if(mDevHeatmap) { cudaFree(mDevHeatmap); mDevHeatmap = nullptr; }
    if(mDevDesc)    { cudaFree(mDevDesc);    mDevDesc    = nullptr; }
    mBufInputSize = mBufHeatmapSize = mBufDescSize = 0;
}

void SPextractor::runInference(const cv::Mat& img, cv::Mat& heatmap, cv::Mat& desc)
{
    if(!mContext) {
        std::cerr << "SPextractor: TensorRT context not initialized" << std::endl;
        return;
    }

    int H = img.rows;
    int W = img.cols;

    cv::Mat inputImg;
    if(mInputH > 0 && mInputW > 0 && (H != mInputH || W != mInputW)) {
        cv::resize(img, inputImg, cv::Size(mInputW, mInputH));
    } else {
        inputImg = img;
        if(mInputH == 0) { mInputH = H; mInputW = W; }
    }

    cv::Mat floatImg;
    inputImg.convertTo(floatImg, CV_32F, 1.0/255.0);

    int inH = floatImg.rows;
    int inW = floatImg.cols;
    int Hc = inH / 8;
    int Wc = inW / 8;

    allocateBuffers(inH, inW);

    cudaMemcpyAsync(mDevInput, floatImg.ptr<float>(), mBufInputSize,
                    cudaMemcpyHostToDevice, mCudaStream);

    // Set dynamic input shape (required for TRT 10.x explicit batch)
    const char* inputName   = mEngine->getIOTensorName(0);
    const char* heatmapName = mEngine->getIOTensorName(1);
    const char* descName    = mEngine->getIOTensorName(2);

    nvinfer1::Dims4 inputDims{1, 1, inH, inW};
    mContext->setInputShape(inputName, inputDims);

    mContext->setTensorAddress(inputName,   mDevInput);
    mContext->setTensorAddress(heatmapName, mDevHeatmap);
    mContext->setTensorAddress(descName,    mDevDesc);

    if(!mContext->enqueueV3(mCudaStream))
    {
        std::cerr << "SPextractor: enqueueV3 failed" << std::endl;
        return;
    }

    // Heatmap: [1, 65, Hc, Wc] -> softmax -> reshape -> [inH, inW]
    std::vector<float> heatmapData(65 * Hc * Wc);
    cudaMemcpyAsync(heatmapData.data(), mDevHeatmap, mBufHeatmapSize,
                    cudaMemcpyDeviceToHost, mCudaStream);

    // Descriptor: [1, 256, Hc, Wc]
    std::vector<float> descData(256 * Hc * Wc);
    cudaMemcpyAsync(descData.data(), mDevDesc, mBufDescSize,
                    cudaMemcpyDeviceToHost, mCudaStream);

    cudaStreamSynchronize(mCudaStream);

    // Softmax over 65 channels at each cell
    heatmap = cv::Mat::zeros(inH, inW, CV_32F);
    for(int y = 0; y < Hc; y++) {
        for(int x = 0; x < Wc; x++) {
            float maxVal = -1e10;
            for(int c = 0; c < 65; c++) {
                float v = heatmapData[c * Hc * Wc + y * Wc + x];
                if(v > maxVal) maxVal = v;
            }

            float sumExp = 0.0f;
            std::vector<float> expVals(65);
            for(int c = 0; c < 65; c++) {
                expVals[c] = exp(heatmapData[c * Hc * Wc + y * Wc + x] - maxVal);
                sumExp += expVals[c];
            }

            for(int c = 0; c < 64; c++) {
                float prob = expVals[c] / sumExp;
                int dy = c / 8;
                int dx = c % 8;
                int py = y * 8 + dy;
                int px = x * 8 + dx;
                if(py < inH && px < inW)
                    heatmap.at<float>(py, px) = prob;
            }
        }
    }

    desc = cv::Mat(256, Hc * Wc, CV_32F);
    memcpy(desc.ptr<float>(), descData.data(), descData.size() * sizeof(float));
    desc = desc.reshape(0, {256, Hc, Wc});
}

void SPextractor::detect(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints,
                         cv::Mat& descriptors, float threshold)
{
    cv::Mat heatmap, descTensor;
    runInference(img, heatmap, descTensor);

    if(heatmap.empty()) return;

    int H = heatmap.rows;
    int W = heatmap.cols;
    int Hc = H / 8;
    int Wc = W / 8;

    // NMS (3x3) + threshold
    std::vector<cv::KeyPoint> rawKps;
    for(int y = 1; y < H - 1; y++) {
        for(int x = 1; x < W - 1; x++) {
            float score = heatmap.at<float>(y, x);
            if(score < threshold) continue;

            bool isMax = true;
            for(int dy = -1; dy <= 1 && isMax; dy++)
                for(int dx = -1; dx <= 1 && isMax; dx++)
                    if((dy != 0 || dx != 0) && heatmap.at<float>(y+dy, x+dx) >= score)
                        isMax = false;

            if(isMax) {
                cv::KeyPoint kp;
                kp.pt = cv::Point2f((float)x, (float)y);
                kp.response = score;
                kp.size = 8.0f;
                kp.octave = 0;
                rawKps.push_back(kp);
            }
        }
    }

    keypoints = rawKps;

    // Interpolate descriptors at keypoint locations
    if(!keypoints.empty() && !descTensor.empty()) {
        descriptors = cv::Mat(keypoints.size(), SP_DESC_DIM, CV_32F);

        const float* descPtr = descTensor.ptr<float>();

        for(size_t i = 0; i < keypoints.size(); i++) {
            float sx = keypoints[i].pt.x / (float)(W) * (float)(Wc);
            float sy = keypoints[i].pt.y / (float)(H) * (float)(Hc);

            int ix = std::min(std::max((int)sx, 0), Wc - 1);
            int iy = std::min(std::max((int)sy, 0), Hc - 1);

            float norm = 0.0f;
            for(int c = 0; c < SP_DESC_DIM; c++) {
                float val = descPtr[c * Hc * Wc + iy * Wc + ix];
                descriptors.at<float>(i, c) = val;
                norm += val * val;
            }

            norm = sqrt(norm);
            if(norm > 1e-6) {
                for(int c = 0; c < SP_DESC_DIM; c++)
                    descriptors.at<float>(i, c) /= norm;
            }
        }
    }
}

vector<cv::KeyPoint> SPextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys,
                                                     const int &minX, const int &maxX,
                                                     const int &minY, const int &maxY,
                                                     const int &N, const int &level)
{
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));
    const float hX = static_cast<float>(maxX-minX)/nIni;

    list<ExtractorNode> lNodes;
    vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for(int i=0; i<nIni; i++)
    {
        ExtractorNode ni;
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
        ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();
    while(lit!=lNodes.end())
    {
        if(lit->vKeys.size()==1)
        {
            lit->bNoMore=true;
            lit++;
        }
        else if(lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;
    int iteration = 0;
    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size()*4);

    while(!bFinish)
    {
        iteration++;
        int prevSize = lNodes.size();
        lit = lNodes.begin();
        int nToExpand = 0;
        vSizeAndPointerToNode.clear();

        while(lit!=lNodes.end())
        {
            if(lit->bNoMore)
            {
                lit++;
                continue;
            }
            else
            {
                ExtractorNode n1,n2,n3,n4;
                lit->DivideNode(n1,n2,n3,n4);

                if(n1.vKeys.size()>0) { lNodes.push_front(n1); if(n1.vKeys.size()>1){ nToExpand++; vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front())); lNodes.front().lit = lNodes.begin(); }}
                if(n2.vKeys.size()>0) { lNodes.push_front(n2); if(n2.vKeys.size()>1){ nToExpand++; vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front())); lNodes.front().lit = lNodes.begin(); }}
                if(n3.vKeys.size()>0) { lNodes.push_front(n3); if(n3.vKeys.size()>1){ nToExpand++; vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front())); lNodes.front().lit = lNodes.begin(); }}
                if(n4.vKeys.size()>0) { lNodes.push_front(n4); if(n4.vKeys.size()>1){ nToExpand++; vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front())); lNodes.front().lit = lNodes.begin(); }}

                lit=lNodes.erase(lit);
                continue;
            }
        }

        if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
        {
            bFinish = true;
        }
        else if(((int)lNodes.size()+nToExpand*3)>N)
        {
            while(!bFinish)
            {
                prevSize = lNodes.size();
                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    if(n1.vKeys.size()>0) { lNodes.push_front(n1); if(n1.vKeys.size()>1){ vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front())); lNodes.front().lit = lNodes.begin(); }}
                    if(n2.vKeys.size()>0) { lNodes.push_front(n2); if(n2.vKeys.size()>1){ vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front())); lNodes.front().lit = lNodes.begin(); }}
                    if(n3.vKeys.size()>0) { lNodes.push_front(n3); if(n3.vKeys.size()>1){ vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front())); lNodes.front().lit = lNodes.begin(); }}
                    if(n4.vKeys.size()>0) { lNodes.push_front(n4); if(n4.vKeys.size()>1){ vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front())); lNodes.front().lit = lNodes.begin(); }}

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if((int)lNodes.size()>=N)
                        break;
                }

                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;
            }
        }
    }

    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures);
    for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

void SPextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint>>& allKeypoints, cv::Mat &_desc)
{
    allKeypoints.resize(nlevels);

    vector<cv::Mat> allDescriptors(nlevels);

    for(int level = 0; level < nlevels; level++)
    {
        const int minBorderX = EDGE_THRESHOLD-3;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD+3;
        const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD+3;

        vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(nfeatures*10);

        cv::Mat levelDesc;

        float threshold = iniThFAST;
        detect(mvImagePyramid[level], vToDistributeKeys, levelDesc, threshold);

        if(vToDistributeKeys.empty()) {
            threshold = minThFAST;
            detect(mvImagePyramid[level], vToDistributeKeys, levelDesc, threshold);
        }

        vector<cv::KeyPoint> filteredKeys;
        cv::Mat filteredDesc;
        for(size_t i = 0; i < vToDistributeKeys.size(); i++) {
            if(vToDistributeKeys[i].pt.x >= minBorderX && vToDistributeKeys[i].pt.x < maxBorderX &&
               vToDistributeKeys[i].pt.y >= minBorderY && vToDistributeKeys[i].pt.y < maxBorderY) {
                filteredKeys.push_back(vToDistributeKeys[i]);
                if(!levelDesc.empty())
                    filteredDesc.push_back(levelDesc.row(i));
            }
        }

        vector<KeyPoint> distributedKeys = DistributeOctTree(filteredKeys,
                                                              minBorderX, maxBorderX,
                                                              minBorderY, maxBorderY,
                                                              mnFeaturesPerLevel[level], level);

        cv::Mat distributedDesc(distributedKeys.size(), SP_DESC_DIM, CV_32F);
        for(size_t i = 0; i < distributedKeys.size(); i++) {
            for(size_t j = 0; j < filteredKeys.size(); j++) {
                if(abs(distributedKeys[i].pt.x - filteredKeys[j].pt.x) < 0.5f &&
                   abs(distributedKeys[i].pt.y - filteredKeys[j].pt.y) < 0.5f) {
                    filteredDesc.row(j).copyTo(distributedDesc.row(i));
                    break;
                }
            }
        }

        const float scaleFactor_level = mvScaleFactor[level];
        const int nkps = distributedKeys.size();
        for(int i = 0; i < nkps; i++) {
            distributedKeys[i].pt.x += minBorderX;
            distributedKeys[i].pt.y += minBorderY;
            distributedKeys[i].pt *= scaleFactor_level;
            distributedKeys[i].octave = level;
            distributedKeys[i].size = PATCH_SIZE * scaleFactor_level;
        }

        allKeypoints[level] = distributedKeys;
        allDescriptors[level] = distributedDesc;
    }

    int nkeypoints = 0;
    for(int level = 0; level < nlevels; level++)
        nkeypoints += (int)allKeypoints[level].size();

    if(nkeypoints == 0) {
        _desc.release();
        return;
    }

    _desc.create(nkeypoints, SP_DESC_DIM, CV_32F);

    int offset = 0;
    for(int level = 0; level < nlevels; level++) {
        int nkeypointsLevel = (int)allKeypoints[level].size();
        if(nkeypointsLevel == 0) continue;

        cv::Mat descLevel = allDescriptors[level];
        if(!descLevel.empty() && descLevel.rows == nkeypointsLevel) {
            descLevel.copyTo(_desc.rowRange(offset, offset + nkeypointsLevel));
        }
        offset += nkeypointsLevel;
    }
}

void SPextractor::ComputePyramid(cv::Mat image)
{
    for(int level = 0; level < nlevels; level++)
    {
        float scale = mvInvScaleFactor[level];
        cv::Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
        cv::Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
        cv::Mat temp(wholeSize, image.type()), masktemp;
        mvImagePyramid[level] = temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        if( level != 0 )
        {
            resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, cv::INTER_LINEAR);

            copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           cv::BORDER_REFLECT_101+cv::BORDER_ISOLATED);
        }
        else
        {
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           cv::BORDER_REFLECT_101);
        }
    }
}

int SPextractor::operator()(InputArray _image, InputArray _mask,
                            vector<KeyPoint>& _keypoints,
                            OutputArray _descriptors, vector<int> &vLappingArea)
{
    if(_image.empty())
        return -1;

    Mat image = _image.getMat();
    assert(image.type() == CV_8UC1 );

    ComputePyramid(image);

    vector<vector<KeyPoint>> allKeypoints;
    cv::Mat descriptors;
    ComputeKeyPointsOctTree(allKeypoints, descriptors);

    int nkeypoints = 0;
    for(int level = 0; level < nlevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();

    if(nkeypoints == 0) {
        _descriptors.release();
    } else {
        _descriptors.create(nkeypoints, SP_DESC_DIM, CV_32F);
        descriptors.copyTo(_descriptors.getMat());
    }

    _keypoints.clear();
    _keypoints.reserve(nkeypoints);

    int offset = 0;
    int monoIndex = 0;
    for(int level = 0; level < nlevels; level++) {
        vector<KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if(nkeypointsLevel == 0) continue;

        for(int i = 0; i < nkeypointsLevel; i++) {
            _keypoints.push_back(keypoints[i]);
        }

        offset += nkeypointsLevel;
    }

    if(vLappingArea.size() >= 2) {
        monoIndex = _keypoints.size();
    }

    return monoIndex;
}

} // namespace ORB_SLAM3
