/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "Viewer.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include <pangolin/pangolin.h>

#include <mutex>
#include <sstream>
#include <iomanip>

namespace ORB_SLAM3
{

Viewer::Viewer(System* pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking, const string &strSettingPath, Settings* settings):
    both(false), mpSystem(pSystem), mpFrameDrawer(pFrameDrawer),mpMapDrawer(pMapDrawer), mpTracker(pTracking),
    mbFinishRequested(false), mbFinished(true), mbStopped(true), mbStopRequested(false)
{
    if(settings){
        newParameterLoader(settings);
    }
    else{

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        bool is_correct = ParseViewerParamFile(fSettings);

        if(!is_correct)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }

    mbStopTrack = false;
}

void Viewer::newParameterLoader(Settings *settings) {
    mImageViewerScale = 1.f;

    float fps = settings->fps();
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    cv::Size imSize = settings->newImSize();
    mImageHeight = imSize.height;
    mImageWidth = imSize.width;

    mImageViewerScale = settings->imageViewerScale();
    mViewpointX = settings->viewPointX();
    mViewpointY = settings->viewPointY();
    mViewpointZ = settings->viewPointZ();
    mViewpointF = settings->viewPointF();
}

bool Viewer::ParseViewerParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;
    mImageViewerScale = 1.f;

    float fps = fSettings["Camera.fps"];
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    cv::FileNode node = fSettings["Camera.width"];
    if(!node.empty())
    {
        mImageWidth = node.real();
    }
    else
    {
        std::cerr << "*Camera.width parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Camera.height"];
    if(!node.empty())
    {
        mImageHeight = node.real();
    }
    else
    {
        std::cerr << "*Camera.height parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.imageViewScale"];
    if(!node.empty())
    {
        mImageViewerScale = node.real();
    }

    node = fSettings["Viewer.ViewpointX"];
    if(!node.empty())
    {
        mViewpointX = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointX parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointY"];
    if(!node.empty())
    {
        mViewpointY = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointY parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointZ"];
    if(!node.empty())
    {
        mViewpointZ = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointZ parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.ViewpointF"];
    if(!node.empty())
    {
        mViewpointF = node.real();
    }
    else
    {
        std::cerr << "*Viewer.ViewpointF parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    return !b_miss_params;
}

void Viewer::Run()
{
    mbFinished = false;
    mbStopped = false;

    pangolin::CreateWindowAndBind("ORB-SLAM3: Map Viewer",1024,768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",false,true);
    pangolin::Var<bool> menuCamView("menu.Camera View",false,false);
    pangolin::Var<bool> menuTopView("menu.Top View",false,false);
    // pangolin::Var<bool> menuSideView("menu.Side View",false,false);
    pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph",false,true);
    pangolin::Var<bool> menuShowInertialGraph("menu.Show Inertial Graph",true,true);
    pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode",false,true);
    pangolin::Var<bool> menuReset("menu.Reset",false,false);
    pangolin::Var<bool> menuStop("menu.Stop",false,false);
    pangolin::Var<bool> menuSaveVelocity("menu.Save Vel",false,false);
    pangolin::Var<bool> menuStepByStep("menu.Step By Step",false,true);  // false, true
    pangolin::Var<bool> menuStep("menu.Step",false,false);

    pangolin::Var<bool> menuShowOptLba("menu.Show LBA opt", false, true);
    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
                );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc, Twr;
    Twc.SetIdentity();
    pangolin::OpenGlMatrix Ow; // Oriented with g in the z axis
    Ow.SetIdentity();
    cv::namedWindow("ORB-SLAM3: Current Frame");

    bool bFollow = true;
    bool bLocalizationMode = false;
    bool bStepByStep = false;
    bool bCameraView = true;

    if(mpTracker->mSensor == mpSystem->MONOCULAR || mpTracker->mSensor == mpSystem->STEREO || mpTracker->mSensor == mpSystem->RGBD)
    {
        menuShowGraph = true;
    }

    float trackedImageScale = mpTracker->GetImageScale();

    // Console print: body/camera axes in SLAM (world) frame, every N viewer frames
    static int s_viewerFrameCount = 0;
    const int kAxesPrintInterval = 30;

    cout << "Starting the Viewer" << endl;
    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc,Ow);

        if(mbStopTrack)
        {
            menuStepByStep = true;
            mbStopTrack = false;
        }

        if(menuFollowCamera && bFollow)
        {
            if(bCameraView)
                s_cam.Follow(Twc);
            else
                s_cam.Follow(Ow);
        }
        else if(menuFollowCamera && !bFollow)
        {
            if(bCameraView)
            {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
                s_cam.Follow(Twc);
            }
            else
            {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,0.01,10, 0,0,0,0.0,0.0, 1.0));
                s_cam.Follow(Ow);
            }
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }

        if(menuCamView)
        {
            menuCamView = false;
            bCameraView = true;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            s_cam.Follow(Twc);
        }

        if(menuTopView && mpMapDrawer->mpAtlas->isImuInitialized())
        {
            menuTopView = false;
            bCameraView = false;
            s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,10000));
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,0.01,50, 0,0,0,0.0,0.0, 1.0));
            s_cam.Follow(Ow);
        }

        if(menuLocalizationMode && !bLocalizationMode)
        {
            mpSystem->ActivateLocalizationMode();
            bLocalizationMode = true;
        }
        else if(!menuLocalizationMode && bLocalizationMode)
        {
            mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
        }

        if(menuStepByStep && !bStepByStep)
        {
            //cout << "Viewer: step by step" << endl;
            mpTracker->SetStepByStep(true);
            bStepByStep = true;
        }
        else if(!menuStepByStep && bStepByStep)
        {
            mpTracker->SetStepByStep(false);
            bStepByStep = false;
        }

        if(menuStep)
        {
            mpTracker->mbStep = true;
            menuStep = false;
        }


        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,1.0f);
        mpMapDrawer->DrawCurrentCamera(Twc);
        if(menuShowKeyFrames || menuShowGraph || menuShowInertialGraph || menuShowOptLba)
            mpMapDrawer->DrawKeyFrames(menuShowKeyFrames,menuShowGraph, menuShowInertialGraph, menuShowOptLba);
        if(menuShowPoints)
            mpMapDrawer->DrawMapPoints();

        pangolin::FinishFrame();

        cv::Mat toShow;
        cv::Mat im = mpFrameDrawer->DrawFrame(trackedImageScale);

        if(both){
            cv::Mat imRight = mpFrameDrawer->DrawRightFrame(trackedImageScale);
            cv::hconcat(im,imRight,toShow);
        }
        else{
            toShow = im;
        }

        if(mImageViewerScale != 1.f)
        {
            int width = toShow.cols * mImageViewerScale;
            int height = toShow.rows * mImageViewerScale;
            cv::resize(toShow, toShow, cv::Size(width, height));
        }

        // On-screen position and velocity (SLAM coords). Smoothed to reduce flicker; always show when pose valid.
        static Eigen::Vector3f s_vSlam(0, 0, 0), s_vImu(0, 0, 0);
        static bool s_velSlamInitialized = false, s_velImuInitialized = false;
        const float kVelSmoothAlpha = 0.88f; // blend: display = alpha*prev + (1-alpha)*raw

        if (!toShow.empty() && mpTracker->mCurrentFrame.HasPose())
        {
            Eigen::Vector3f pos = mpTracker->mCurrentFrame.GetCameraCenter();
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(3);
            ss << "Pos: " << pos(0) << " " << pos(1) << " " << pos(2);
            int y = 22, dy = 22;
            cv::putText(toShow, ss.str(), cv::Point(8, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            y += dy;

            // Raw Vel(SLAM): prefer per-frame (finite-diff or current frame) for real-time; fallback to ref KF
            Eigen::Vector3f vRawSlam(0, 0, 0);
            bool haveRawSlam = false;
            if (mpTracker->mSensor == System::MONOCULAR || mpTracker->mSensor == System::STEREO || mpTracker->mSensor == System::RGBD)
            {
                if (mpTracker->mLastFrame.HasPose())
                {
                    double dt = mpTracker->mCurrentFrame.mTimeStamp - mpTracker->mLastFrame.mTimeStamp;
                    if (dt > 1e-6)
                    {
                        Eigen::Vector3f posLast = mpTracker->mLastFrame.GetCameraCenter();
                        vRawSlam = (pos - posLast) / static_cast<float>(dt);
                        haveRawSlam = true;
                    }
                }
            }
            if (!haveRawSlam && mpTracker->mCurrentFrame.HasVelocity())
                { vRawSlam = mpTracker->mCurrentFrame.GetVelocity(); haveRawSlam = true; }
            if (!haveRawSlam)
            {
                KeyFrame* pRefKF = mpTracker->mCurrentFrame.mpReferenceKF;
                if (pRefKF && !pRefKF->isBad())
                {
                    Eigen::Vector3f v = pRefKF->GetVelocity();
                    if (v.norm() >= 1e-6f) { vRawSlam = v; haveRawSlam = true; }
                }
            }

            if (haveRawSlam)
            {
                if (!s_velSlamInitialized) { s_vSlam = vRawSlam; s_velSlamInitialized = true; }
                else s_vSlam = kVelSmoothAlpha * s_vSlam + (1.f - kVelSmoothAlpha) * vRawSlam;
            }
            // Always show Vel(SLAM) when we have pose (use smoothed or 0)
            ss.str(""); ss << std::setprecision(2) << "Vel(SLAM): " << s_vSlam(0) << " " << s_vSlam(1) << " " << s_vSlam(2);
            cv::putText(toShow, ss.str(), cv::Point(8, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
            y += dy;

            bool haveRawImu = (mpTracker->mSensor == System::IMU_MONOCULAR || mpTracker->mSensor == System::IMU_STEREO || mpTracker->mSensor == System::IMU_RGBD) && mpTracker->GetLastImuPredictedValid();
            if (haveRawImu)
            {
                Eigen::Vector3f vimu = mpTracker->GetLastImuPredictedVelocity();
                if (!s_velImuInitialized) { s_vImu = vimu; s_velImuInitialized = true; }
                else s_vImu = kVelSmoothAlpha * s_vImu + (1.f - kVelSmoothAlpha) * vimu;
                ss.str(""); ss << std::setprecision(2) << "Vel(IMU): " << s_vImu(0) << " " << s_vImu(1) << " " << s_vImu(2);
                cv::putText(toShow, ss.str(), cv::Point(8, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 200, 0), 1, cv::LINE_AA);
            }
        }
        else
        {
            s_velSlamInitialized = false;
            s_velImuInitialized = false;
        }

        // Body/camera axes in SLAM (world) frame: computed every frame, shown on-screen for real-time
        if (!toShow.empty() && mpTracker->mCurrentFrame.HasPose())
        {
            Eigen::Matrix3f R;
            bool isBody = (mpTracker->mSensor == System::IMU_MONOCULAR || mpTracker->mSensor == System::IMU_STEREO || mpTracker->mSensor == System::IMU_RGBD);
            if (isBody)
                R = mpTracker->mCurrentFrame.GetImuRotation();
            else
                R = mpTracker->mCurrentFrame.GetRwc();
            int yAxes = 78; // below Pos + Vel lines
            std::ostringstream ssAxes;
            ssAxes << std::fixed << std::setprecision(2);
            ssAxes << "Xw: " << R(0,0) << " " << R(1,0) << " " << R(2,0);
            cv::putText(toShow, ssAxes.str(), cv::Point(8, yAxes), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
            yAxes += 18;
            ssAxes.str(""); ssAxes << "Yw: " << R(0,1) << " " << R(1,1) << " " << R(2,1);
            cv::putText(toShow, ssAxes.str(), cv::Point(8, yAxes), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
            yAxes += 18;
            ssAxes.str(""); ssAxes << "Zw: " << R(0,2) << " " << R(1,2) << " " << R(2,2);
            cv::putText(toShow, ssAxes.str(), cv::Point(8, yAxes), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        }

        // Console: axes, velocity (s_vSlam), point cloud every kAxesPrintInterval — all from current frame F like R
        if (mpTracker->mCurrentFrame.HasPose() && (++s_viewerFrameCount % kAxesPrintInterval == 0))
        {
            const Frame& F = mpTracker->mCurrentFrame;

            // R: camera/body axes in SLAM (world) frame — from F like elsewhere
            Eigen::Matrix3f R;
            bool isBody = (mpTracker->mSensor == System::IMU_MONOCULAR || mpTracker->mSensor == System::IMU_STEREO || mpTracker->mSensor == System::IMU_RGBD);
            if (isBody)
                R = F.GetImuRotation();
            else
                R = F.GetRwc();
            std::cout << "[SLAM] " << (isBody ? "Body" : "Camera") << " axes in world:" << std::endl;
            std::cout << "  X: " << R(0,0) << " " << R(1,0) << " " << R(2,0) << std::endl;
            std::cout << "  Y: " << R(0,1) << " " << R(1,1) << " " << R(2,1) << std::endl;
            std::cout << "  Z: " << R(0,2) << " " << R(1,2) << " " << R(2,2) << std::endl;

            // s_vSlam: velocity in SLAM (world) computed here from F (and mLastFrame) like R, then update smoothed display
            Eigen::Vector3f Vw(0, 0, 0);
            bool haveVw = false;
            if (mpTracker->mSensor == System::MONOCULAR || mpTracker->mSensor == System::STEREO || mpTracker->mSensor == System::RGBD)
            {
                if (mpTracker->mLastFrame.HasPose())
                {
                    double dt = F.mTimeStamp - mpTracker->mLastFrame.mTimeStamp;
                    if (dt > 1e-6)
                    {
                        Vw = (F.GetCameraCenter() - mpTracker->mLastFrame.GetCameraCenter()) / static_cast<float>(dt);
                        haveVw = true;
                    }
                }
            }
            else if (F.HasVelocity())
                { Vw = F.GetVelocity(); haveVw = true; }
            if (!haveVw && F.mpReferenceKF && !F.mpReferenceKF->isBad())
            {
                Eigen::Vector3f vRef = F.mpReferenceKF->GetVelocity();
                if (vRef.norm() >= 1e-6f) { Vw = vRef; haveVw = true; }
            }
            if (haveVw)
            {
                if (!s_velSlamInitialized) { s_vSlam = Vw; s_velSlamInitialized = true; }
                else s_vSlam = 0.88f * s_vSlam + 0.12f * Vw;
                std::cout << "[SLAM] Velocity in world (s_vSlam): " << s_vSlam(0) << " " << s_vSlam(1) << " " << s_vSlam(2) << std::endl;
            }

            // Point cloud: map points seen in current frame — camera frame and SLAM (world) frame
            const Sophus::SE3f Tcw = F.GetPose();
            const int kMaxPointsToPrint = 5;
            int nPrinted = 0;
            for (size_t i = 0; i < F.mvpMapPoints.size() && nPrinted < kMaxPointsToPrint; ++i)
            {
                if (!F.mvpMapPoints[i] || F.mvbOutlier[i]) continue;
                MapPoint* pMP = F.mvpMapPoints[i];
                if (pMP->isBad()) continue;
                Eigen::Vector3f Pw = pMP->GetWorldPos();
                Eigen::Vector3f Pc = Tcw.rotationMatrix() * Pw + Tcw.translation();
                std::cout << "[SLAM] Point " << nPrinted << " camera: " << Pc(0) << " " << Pc(1) << " " << Pc(2)
                          << "  world: " << Pw(0) << " " << Pw(1) << " " << Pw(2) << std::endl;
                ++nPrinted;
            }
            if (nPrinted > 0)
                std::cout << "[SLAM] (showing " << nPrinted << " points; total in frame: " << F.N << ")" << std::endl;
        }

        cv::imshow("ORB-SLAM3: Current Frame",toShow);
        cv::waitKey(mT);

        if(menuReset)
        {
            menuShowGraph = true;
            menuShowInertialGraph = true;
            menuShowKeyFrames = true;
            menuShowPoints = true;
            menuLocalizationMode = false;
            if(bLocalizationMode)
                mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
            bFollow = true;
            menuFollowCamera = true;
            mpSystem->ResetActiveMap();
            menuReset = false;
        }

        if(menuSaveVelocity)
        {
            std::cout << "[Viewer] Save Vel: writing files ..." << std::endl;
            mpSystem->SaveTrajectoryEuRoCWithVelocity("CameraTrajectoryWithVelocity.txt");
            mpSystem->SaveKeyFrameTrajectoryEuRoCWithVelocity("KeyFrameTrajectoryWithVelocity.txt");
            std::cout << "[Viewer] Save Vel: done. (CameraTrajectoryWithVelocity.txt, KeyFrameTrajectoryWithVelocity.txt)" << std::endl;
            menuSaveVelocity = false;
        }

        if(menuStop)
        {
            if(bLocalizationMode)
                mpSystem->DeactivateLocalizationMode();

            // Stop all threads
            mpSystem->Shutdown();

            // Save camera trajectory
            mpSystem->SaveTrajectoryEuRoC("CameraTrajectory.txt");
            mpSystem->SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
            menuStop = false;
        }

        if(Stop())
        {
            while(isStopped())
            {
                usleep(3000);
            }
        }

        if(CheckFinish())
            break;
    }

    SetFinish();
}

void Viewer::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Viewer::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(!mbStopped)
        mbStopRequested = true;
}

bool Viewer::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);

    if(mbFinishRequested)
        return false;
    else if(mbStopRequested)
    {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }

    return false;

}

void Viewer::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}

/*void Viewer::SetTrackingPause()
{
    mbStopTrack = true;
}*/

}
