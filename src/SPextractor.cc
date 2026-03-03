#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <onnxruntime_cxx_api.h>

#include "SPextractor.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM3
{

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;
const int SP_DESC_DIM = 256;

const float SP_THRESHOLD_HIGH = 0.015f;
const float SP_THRESHOLD_LOW  = 0.005f;


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

SPextractor::SPextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                         float _iniThFAST, float _minThFAST,
                         const std::string& modelPath)
    : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
      iniThFAST(_iniThFAST), minThFAST(_minThFAST), mModelPath(modelPath)
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

    loadModel(mModelPath);
}

void SPextractor::loadModel(const std::string& modelPath)
{
    try {
        mEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");

        mSessionOptions.SetIntraOpNumThreads(4);
        mSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA
        try {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = mDeviceId;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnDefaultConvAlgoSearch;
            cuda_options.gpu_mem_limit = SIZE_MAX;
            cuda_options.do_copy_in_default_stream = 1;
            mSessionOptions.AppendExecutionProvider_CUDA(cuda_options);
            mUseCUDA = true;
            std::cout << "SPextractor: CUDA execution provider appended (device "
                      << mDeviceId << ")" << std::endl;
        }
        catch(const Ort::Exception& e) {
            std::cerr << "SPextractor: CUDA provider failed, falling back to CPU: "
                      << e.what() << std::endl;
            mUseCUDA = false;
        }
#else
        mUseCUDA = false;
#endif

#ifdef _WIN32
        std::wstring wpath(modelPath.begin(), modelPath.end());
        mSession = std::make_unique<Ort::Session>(*mEnv, wpath.c_str(), mSessionOptions);
#else
        mSession = std::make_unique<Ort::Session>(*mEnv, modelPath.c_str(), mSessionOptions);
#endif

        Ort::AllocatorWithDefaultOptions allocator;

        size_t numInputs = mSession->GetInputCount();
        size_t numOutputs = mSession->GetOutputCount();

        std::cout << "SPextractor: ONNX model has " << numInputs << " inputs and "
                  << numOutputs << " outputs:" << std::endl;

        mInputNamesStr.clear();
        mOutputNamesStr.clear();

        for(size_t i = 0; i < numInputs; i++)
        {
            auto namePtr = mSession->GetInputNameAllocated(i, allocator);
            std::string name(namePtr.get());
            mInputNamesStr.push_back(name);

            auto typeInfo = mSession->GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            auto shape = tensorInfo.GetShape();

            std::cout << "  Input[" << i << "] \"" << name << "\" shape=[";
            for(size_t d = 0; d < shape.size(); d++)
                std::cout << (d ? "," : "") << shape[d];
            std::cout << "]" << std::endl;

            if(i == 0 && shape.size() == 4) {
                mInputH = (shape[2] > 0) ? static_cast<int>(shape[2]) : 0;
                mInputW = (shape[3] > 0) ? static_cast<int>(shape[3]) : 0;
                if(mInputH <= 0 || mInputW <= 0)
                    mDynamicShape = true;
            }
        }

        for(size_t i = 0; i < numOutputs; i++)
        {
            auto namePtr = mSession->GetOutputNameAllocated(i, allocator);
            std::string name(namePtr.get());
            mOutputNamesStr.push_back(name);

            auto typeInfo = mSession->GetOutputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            auto shape = tensorInfo.GetShape();

            std::cout << "  Output[" << i << "] \"" << name << "\" shape=[";
            for(size_t d = 0; d < shape.size(); d++)
                std::cout << (d ? "," : "") << shape[d];
            std::cout << "]" << std::endl;
        }

        mInputNames.clear();
        for(auto& s : mInputNamesStr)
            mInputNames.push_back(s.c_str());
        mOutputNames.clear();
        for(auto& s : mOutputNamesStr)
            mOutputNames.push_back(s.c_str());

        std::cout << "SPextractor: ONNX Runtime model loaded ("
                  << (mUseCUDA ? "CUDA" : "CPU") << ")."
                  << " Input: " << mInputH << "x" << mInputW
                  << (mDynamicShape ? " (dynamic)" : " (fixed)")
                  << std::endl;
    }
    catch(const Ort::Exception& e) {
        std::cerr << "SPextractor: Failed to load ONNX model: " << e.what() << std::endl;
    }
}

void SPextractor::runInference(const cv::Mat& img, cv::Mat& heatmap, cv::Mat& desc)
{
    if(!mSession) {
        std::cerr << "SPextractor: ONNX session not initialized" << std::endl;
        return;
    }

    int origH = img.rows;
    int origW = img.cols;

    // Letterbox: scale to fit model input while preserving aspect ratio, pad the rest
    mPadTop = 0;
    mPadLeft = 0;
    mLetterboxScaleX = 1.0f;
    mLetterboxScaleY = 1.0f;
    mResizedW = origW;
    mResizedH = origH;
    cv::Mat inputImg;
    if(!mDynamicShape && mInputH > 0 && mInputW > 0 && (origH != mInputH || origW != mInputW)) {
        float uniformScale = std::min((float)mInputW / origW, (float)mInputH / origH);
        int newW = ((int)(origW * uniformScale) / 8) * 8;
        int newH = ((int)(origH * uniformScale) / 8) * 8;
        if(newW < 8) newW = 8;
        if(newH < 8) newH = 8;

        mLetterboxScaleX = (float)newW / origW;
        mLetterboxScaleY = (float)newH / origH;
        mResizedW = newW;
        mResizedH = newH;

        mPadLeft = ((mInputW - newW) / 2 / 8) * 8;
        mPadTop  = ((mInputH - newH) / 2 / 8) * 8;

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(newW, newH));
        inputImg = cv::Mat::zeros(mInputH, mInputW, img.type());
        resized.copyTo(inputImg(cv::Rect(mPadLeft, mPadTop, newW, newH)));
    } else {
        inputImg = img;
    }

    cv::Mat floatImg;
    inputImg.convertTo(floatImg, CV_32F, 1.0/255.0);

    int inH = floatImg.rows;
    int inW = floatImg.cols;
    int Hc = inH / 8;
    int Wc = inW / 8;

    std::vector<int64_t> inputShape = {1, 1, inH, inW};
    size_t inputTensorSize = 1 * 1 * inH * inW;

    std::vector<float> inputTensorValues(inputTensorSize);
    if(floatImg.isContinuous()) {
        memcpy(inputTensorValues.data(), floatImg.ptr<float>(), inputTensorSize * sizeof(float));
    } else {
        for(int r = 0; r < inH; r++)
            memcpy(inputTensorValues.data() + r * inW, floatImg.ptr<float>(r), inW * sizeof(float));
    }

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputShape.data(), inputShape.size());

    try {
        auto outputTensors = mSession->Run(
            Ort::RunOptions{nullptr},
            mInputNames.data(), &inputTensor, 1,
            mOutputNames.data(), mOutputNames.size());

        int semiIdx = -1, descIdx = -1;
        for(size_t i = 0; i < outputTensors.size(); i++) {
            auto info = outputTensors[i].GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();
            if(shape.size() >= 2) {
                if(shape[1] == 65)
                    semiIdx = static_cast<int>(i);
                else if(shape[1] == 256)
                    descIdx = static_cast<int>(i);
            }
        }

        if(semiIdx < 0 || descIdx < 0) {
            if(outputTensors.size() >= 2) {
                semiIdx = 0;
                descIdx = 1;
            } else {
                std::cerr << "SPextractor: Cannot identify output tensors" << std::endl;
                return;
            }
        }

        // Verify and log actual tensor shapes (once)
        static bool shapeLogged = false;
        if(!shapeLogged) {
            for(size_t i = 0; i < outputTensors.size(); i++) {
                auto shape = outputTensors[i].GetTensorTypeAndShapeInfo().GetShape();
                std::cout << "[SPextractor] output[" << i << "] shape=[";
                for(size_t s = 0; s < shape.size(); s++)
                    std::cout << (s>0?",":"") << shape[s];
                std::cout << "]" << std::endl;
            }
            shapeLogged = true;
        }

        // Use ACTUAL descriptor tensor spatial dimensions instead of assuming Hc/Wc
        auto descShape = outputTensors[descIdx].GetTensorTypeAndShapeInfo().GetShape();
        int actualDescHc = (descShape.size() >= 3) ? descShape[descShape.size()-2] : Hc;
        int actualDescWc = (descShape.size() >= 3) ? descShape[descShape.size()-1] : Wc;

        const float* semiData = outputTensors[semiIdx].GetTensorData<float>();

        bool needSoftmax = false;
        {
            float channelSum = 0.0f;
            bool hasNegative = false;
            for(int c = 0; c < 65; c++) {
                float v = semiData[c * Hc * Wc];
                channelSum += v;
                if(v < 0.0f) hasNegative = true;
            }
            needSoftmax = hasNegative || (channelSum < 0.5f) || (channelSum > 1.5f);
        }

        // Build heatmap at MODEL resolution (inH x inW)
        // Keypoint extraction and descriptor sampling both happen in model space
        heatmap = cv::Mat::zeros(inH, inW, CV_32F);
        for(int y = 0; y < Hc; y++) {
            for(int x = 0; x < Wc; x++) {
                std::vector<float> probs(64);

                if(needSoftmax) {
                    float maxVal = -1e10f;
                    for(int c = 0; c < 65; c++) {
                        float v = semiData[c * Hc * Wc + y * Wc + x];
                        if(v > maxVal) maxVal = v;
                    }
                    float sumExp = 0.0f;
                    std::vector<float> expVals(65);
                    for(int c = 0; c < 65; c++) {
                        expVals[c] = exp(semiData[c * Hc * Wc + y * Wc + x] - maxVal);
                        sumExp += expVals[c];
                    }
                    for(int c = 0; c < 64; c++)
                        probs[c] = expVals[c] / sumExp;
                } else {
                    for(int c = 0; c < 64; c++)
                        probs[c] = semiData[c * Hc * Wc + y * Wc + x];
                }

                for(int c = 0; c < 64; c++) {
                    int dy = c / 8;
                    int dx = c % 8;
                    int px = x * 8 + dx;
                    int py = y * 8 + dy;
                    if(py < inH && px < inW)
                        heatmap.at<float>(py, px) = probs[c];
                }
            }
        }

        const float* descData = outputTensors[descIdx].GetTensorData<float>();
        desc = cv::Mat(256, actualDescHc * actualDescWc, CV_32F);
        memcpy(desc.ptr<float>(), descData, 256 * actualDescHc * actualDescWc * sizeof(float));
        desc = desc.reshape(0, {256, actualDescHc, actualDescWc});
    }
    catch(const Ort::Exception& e) {
        std::cerr << "SPextractor: ONNX inference failed ("
                  << (mUseCUDA ? "CUDA" : "CPU") << "): " << e.what() << std::endl;
    }
}

void SPextractor::detect(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints,
                         cv::Mat& descriptors, float threshold)
{
    cv::Mat heatmap, descTensor;
    runInference(img, heatmap, descTensor);

    if(heatmap.empty()) return;

    int origH = img.rows;
    int origW = img.cols;

    // Heatmap is at MODEL resolution (e.g. 512x512)
    int mH = heatmap.rows;
    int mW = heatmap.cols;

    // Descriptor grid dimensions
    int descHc = descTensor.empty() ? 0 : descTensor.size[1];
    int descWc = descTensor.empty() ? 0 : descTensor.size[2];

    // Only search for keypoints within the non-padded image region
    int roiX0 = mPadLeft + 1;
    int roiY0 = mPadTop + 1;
    int roiX1 = mPadLeft + mResizedW - 1;
    int roiY1 = mPadTop + mResizedH - 1;
    roiX1 = std::min(roiX1, mW - 1);
    roiY1 = std::min(roiY1, mH - 1);

    // Extract keypoints in MODEL pixel space with NMS
    struct ModelKeypoint {
        int mx, my;
        float score;
    };
    std::vector<ModelKeypoint> modelKps;
    for(int y = roiY0; y < roiY1; y++) {
        for(int x = roiX0; x < roiX1; x++) {
            float score = heatmap.at<float>(y, x);
            if(score < threshold) continue;

            bool isMax = true;
            for(int dy = -1; dy <= 1 && isMax; dy++)
                for(int dx = -1; dx <= 1 && isMax; dx++)
                    if((dy != 0 || dx != 0) && heatmap.at<float>(y+dy, x+dx) >= score)
                        isMax = false;

            if(isMax)
                modelKps.push_back({x, y, score});
        }
    }

    if(modelKps.empty()) return;

    const float* descPtr = descTensor.empty() ? nullptr : descTensor.ptr<float>();

    // One-time descriptor grid diagnostic
    static bool descDiagDone = false;
    if(!descDiagDone && descPtr && descHc > 0 && descWc > 0) {
        descDiagDone = true;
        float gMin = 1e9, gMax = -1e9;
        double gSum = 0;
        int gCount = 0;
        // Check a few grid cells in the image region
        for(int gy = mPadTop/8 + 1; gy < (mPadTop + mResizedH)/8 - 1 && gy < descHc; gy += std::max(1, descHc/8)) {
            for(int gx = mPadLeft/8 + 1; gx < (mPadLeft + mResizedW)/8 - 1 && gx < descWc; gx += std::max(1, descWc/8)) {
                float cellNorm = 0;
                for(int c = 0; c < SP_DESC_DIM; c++) {
                    float v = descPtr[c * descHc * descWc + gy * descWc + gx];
                    cellNorm += v * v;
                    if(v < gMin) gMin = v;
                    if(v > gMax) gMax = v;
                }
                cellNorm = sqrt(cellNorm);
                gSum += cellNorm;
                gCount++;
            }
        }
        std::cout << "[SPextractor] desc_grid=" << descWc << "x" << descHc
                  << " value_range=[" << gMin << "," << gMax << "]"
                  << " avg_norm=" << (gCount > 0 ? gSum/gCount : 0)
                  << " image_grid_region=[" << mPadLeft/8 << "-" << (mPadLeft+mResizedW)/8
                  << ", " << mPadTop/8 << "-" << (mPadTop+mResizedH)/8 << "]"
                  << " raw_kps=" << modelKps.size() << std::endl;
    }

    keypoints.resize(modelKps.size());
    descriptors = cv::Mat::zeros(modelKps.size(), SP_DESC_DIM, CV_32F);

    for(size_t i = 0; i < modelKps.size(); i++) {
        int mx = modelKps[i].mx;
        int my = modelKps[i].my;

        // Convert model pixel coords to original image coords
        float origX = ((float)mx - mPadLeft) / mLetterboxScaleX;
        float origY = ((float)my - mPadTop) / mLetterboxScaleY;
        origX = std::max(0.0f, std::min(origX, (float)(origW - 1)));
        origY = std::max(0.0f, std::min(origY, (float)(origH - 1)));

        keypoints[i].pt = cv::Point2f(origX, origY);
        keypoints[i].response = modelKps[i].score;
        keypoints[i].size = 8.0f;
        keypoints[i].octave = 0;

        if(!descPtr) continue;

        // Sample descriptor at MODEL grid position (clean integer-derived coords)
        float sx = (float)mx / 8.0f;
        float sy = (float)my / 8.0f;

        // Bilinear interpolation (matches PyTorch grid_sample)
        int ix0 = (int)floor(sx);
        int iy0 = (int)floor(sy);
        int ix1 = ix0 + 1;
        int iy1 = iy0 + 1;
        float fx = sx - ix0;
        float fy = sy - iy0;

        ix0 = std::max(0, std::min(ix0, descWc - 1));
        ix1 = std::max(0, std::min(ix1, descWc - 1));
        iy0 = std::max(0, std::min(iy0, descHc - 1));
        iy1 = std::max(0, std::min(iy1, descHc - 1));

        float w00 = (1.0f - fx) * (1.0f - fy);
        float w10 = fx * (1.0f - fy);
        float w01 = (1.0f - fx) * fy;
        float w11 = fx * fy;

        float norm = 0.0f;
        for(int c = 0; c < SP_DESC_DIM; c++) {
            float val = w00 * descPtr[c * descHc * descWc + iy0 * descWc + ix0]
                      + w10 * descPtr[c * descHc * descWc + iy0 * descWc + ix1]
                      + w01 * descPtr[c * descHc * descWc + iy1 * descWc + ix0]
                      + w11 * descPtr[c * descHc * descWc + iy1 * descWc + ix1];
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
        int idx = (int)(kp.pt.x/hX);
        if(idx >= nIni) idx = nIni - 1;
        if(idx < 0) idx = 0;
        vpIniNodes[idx]->vKeys.push_back(kp);
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

        float threshold = SP_THRESHOLD_HIGH;
        detect(mvImagePyramid[level], vToDistributeKeys, levelDesc, threshold);

        if(vToDistributeKeys.empty()) {
            threshold = SP_THRESHOLD_LOW;
            detect(mvImagePyramid[level], vToDistributeKeys, levelDesc, threshold);
        }

        // Filter keypoints to valid border region, shift coords for OctTree
        // Use class_id to track original index for descriptor lookup
        vector<cv::KeyPoint> filteredKeys;
        for(size_t i = 0; i < vToDistributeKeys.size(); i++) {
            if(vToDistributeKeys[i].pt.x >= minBorderX && vToDistributeKeys[i].pt.x < maxBorderX &&
               vToDistributeKeys[i].pt.y >= minBorderY && vToDistributeKeys[i].pt.y < maxBorderY) {
                cv::KeyPoint kp = vToDistributeKeys[i];
                kp.pt.x -= minBorderX;
                kp.pt.y -= minBorderY;
                kp.class_id = (int)i;
                filteredKeys.push_back(kp);
            }
        }

        vector<KeyPoint> distributedKeys = DistributeOctTree(filteredKeys,
                                                              0, maxBorderX - minBorderX,
                                                              0, maxBorderY - minBorderY,
                                                              mnFeaturesPerLevel[level], level);

        // Look up descriptors using tracked indices
        cv::Mat distributedDesc = cv::Mat::zeros(distributedKeys.size(), SP_DESC_DIM, CV_32F);
        if(!levelDesc.empty()) {
            for(size_t i = 0; i < distributedKeys.size(); i++) {
                int origIdx = distributedKeys[i].class_id;
                if(origIdx >= 0 && origIdx < levelDesc.rows)
                    levelDesc.row(origIdx).copyTo(distributedDesc.row(i));
            }
        }

        // Restore border offset and scale to original image coordinates
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

    // --- DEBUG: keypoint extraction summary ---
    std::cout << "[SPextractor] image=" << image.cols << "x" << image.rows
              << " model=" << (mDynamicShape ? "dynamic" : std::to_string(mInputW)+"x"+std::to_string(mInputH))
              << " letterbox_scale=" << mLetterboxScaleX << "x" << mLetterboxScaleY << " pad=" << mPadLeft << "," << mPadTop
              << " levels=" << nlevels
              << " total_keypoints=" << nkeypoints;
    if(nkeypoints > 0) {
        float minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
        for(const auto& kp : _keypoints) {
            if(kp.pt.x < minX) minX = kp.pt.x;
            if(kp.pt.x > maxX) maxX = kp.pt.x;
            if(kp.pt.y < minY) minY = kp.pt.y;
            if(kp.pt.y > maxY) maxY = kp.pt.y;
        }
        std::cout << " kp_range_x=[" << minX << "," << maxX << "]"
                  << " kp_range_y=[" << minY << "," << maxY << "]";
    }
    std::cout << std::endl;

    return monoIndex;
}

} // namespace ORB_SLAM3
