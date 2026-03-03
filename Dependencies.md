## List of Known Dependencies
### SuperPoint-SLAM3

In this document we list all the pieces of code included by SuperPoint-SLAM3 and linked libraries which are not property of the original authors.


##### Code in **src** and **include** folders

* *SPextractor.cc*.
SuperPoint feature extractor using TensorRT inference. Keypoint distribution uses octree code adapted from the original ORB-SLAM3 ORBextractor (BSD licensed, from OpenCV).

* *PnPsolver.h, PnPsolver.cc*.
This is a modified version of the epnp.h and epnp.cc of Vincent Lepetit. 
This code can be found in popular BSD licensed computer vision libraries as [OpenCV](https://github.com/Itseez/opencv/blob/master/modules/calib3d/src/epnp.cpp) and [OpenGV](https://github.com/laurentkneip/opengv/blob/master/src/absolute_pose/modules/Epnp.cpp). The original code is FreeBSD.

* *MLPnPsolver.h, MLPnPsolver.cc*.
This is a modified version of the MLPnP of Steffen Urban from [here](https://github.com/urbste/opengv). 
The original code is BSD licensed.

* Function *ORBmatcher::DescriptorDistance* in *ORBmatcher.cc*.
Computes L2 distance between float descriptors using OpenCV norm.

##### Code in Thirdparty folder

* All code in **DBoW3** folder.
[DBoW3](https://github.com/rmsalinas/DBow3) library for bag-of-words place recognition with float descriptors. BSD licensed.

* All code in **g2o** folder.
This is a modified version of [g2o](https://github.com/RainerKuemmerle/g2o). All files included are BSD licensed.

* All code in **Sophus** folder.
This is a modified version of [Sophus](https://github.com/strasdat/Sophus). [MIT license](https://en.wikipedia.org/wiki/MIT_License).

##### Library dependencies 

* **Pangolin (visualization and user interface)**.
[MIT license](https://en.wikipedia.org/wiki/MIT_License).

* **OpenCV**.
BSD license.

* **Eigen3**.
For versions greater than 3.1.1 is MPL2, earlier versions are LGPLv3.

* **CUDA Runtime**.
NVIDIA proprietary license.

* **TensorRT**.
NVIDIA proprietary license.

* **Boost (serialization)**.
Boost Software License.

* **ROS (Optional, only if you build Examples/ROS)**.
BSD license. In the manifest.xml the only declared package dependencies are roscpp, tf, sensor_msgs, image_transport, cv_bridge, which are all BSD licensed.
