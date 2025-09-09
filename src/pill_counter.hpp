#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

namespace pilseg {

    // ----- Data carriers -----
    struct FusionOutputs {
        Mat lumMask;    // 0/255
        Mat chromaMask; // 0/255
        Mat fused;      // 0/255
    };

    struct WSOut {
        Mat segMask;     // 0/255
        Mat markers;     // CV_32S labels
        vector<Rect> boxes;
    };

    double autoClipLimit(const Mat& L8,
                        double clipMin = 1.5, double clipMax = 5.0);

    int autoBlockSize(const Size& sz);

    Mat luminanceMask(const Mat& bgr,
                        const string& mode = "adaptive",
                        Size tileGrid = Size(8,8));

    Mat chromaMagnitudeF32(const Mat& bgr);
    Mat normalizeToU8(const Mat& src32f);

    Mat chromaMask(const Mat& bgr,
                    const string& mode = "otsu"); // "otsu" | "kmeans"

    FusionOutputs fusedMaskForWatershed(const Mat& bgr,
                                        const string& lumMode    = "adaptive",
                                        const string& chromaMode = "otsu");

    Mat watershedSeeds(const Mat& fusedMask);

    Mat makeSeedsFromFused(const Mat& fused, double fgPercentile = 0.65);

    WSOut runWatershed(const Mat& bgr, const Mat& fused,
                    double fgPercentile = 0.65);

} // namespace pilseg