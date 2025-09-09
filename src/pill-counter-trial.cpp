#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int countObjects(const Mat& markers) {
    set<int> uniqueLabels;
    for (int r = 0; r < markers.rows; r++) {
        for (int c = 0; c < markers.cols; c++) {
            int val = markers.at<int>(r, c);
            if (val > 1) { // exclude background (1) and boundaries (-1)
                uniqueLabels.insert(val);
            }
        }
    }
    return (int)uniqueLabels.size();
}

int main() {
    //loading both the images
    Mat red_pill_img = imread("images/red-pills-white-bg.jpg");
    Mat blue_pill_img = imread("images/blue-pills-white-bg.jpg");
    
    Mat rp_gray, bp_gray, rp_th, bp_th;
    cvtColor(red_pill_img, rp_gray, COLOR_BGR2GRAY);
    GaussianBlur(rp_gray, rp_gray, Size(3, 3), 0);
    cvtColor(blue_pill_img, bp_gray, COLOR_BGR2GRAY);
    GaussianBlur(bp_gray, bp_gray, Size(3, 3), 0);

    imshow("Denoised image for red pills", rp_gray);
    imshow("Denoised image for blue pills", bp_gray);

    // Binary mask (pills as white)
    threshold(rp_gray, rp_th, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    threshold(bp_gray, bp_th, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    imshow("Binary mask for red pills", rp_th);
    imshow("Binary mask for blue pills", bp_th);

    // Morphology to remove noise
    Mat kernel = Mat::ones(3, 3, CV_8U);
    morphologyEx(rp_th, rp_th, MORPH_OPEN, kernel, Point(-1,-1), 2);
    morphologyEx(bp_th, bp_th, MORPH_OPEN, kernel, Point(-1,-1), 2);

    // Distance transform
    Mat rp_dist, bp_dist;
    distanceTransform(rp_th, rp_dist, DIST_L2, 5);
    normalize(rp_dist, rp_dist, 0, 1.0, NORM_MINMAX);
    distanceTransform(bp_th, bp_dist, DIST_L2, 5);
    normalize(bp_dist, bp_dist, 0, 1.0, NORM_MINMAX);

    imshow("Distance transformed image for red pills", rp_dist);
    imshow("Distance transformed image for blue pills", bp_dist);

    // Adaptive threshold based on max dist value
    double rp_maxVal, bp_maxVal;
    minMaxLoc(rp_dist, nullptr, &rp_maxVal);
    minMaxLoc(bp_dist, nullptr, &bp_maxVal);

    // setting the max threshold values
    double rp_thresh_val = 0.55 * rp_maxVal;
    double bp_thresh_val = 0.41 * bp_maxVal;

    // Foreground (sure objects)
    Mat rp_sure_fg, bp_sure_fg;
    threshold(rp_dist, rp_sure_fg, rp_thresh_val, 255, 0);
    rp_sure_fg.convertTo(rp_sure_fg, CV_8U);
    threshold(bp_dist, bp_sure_fg, bp_thresh_val, 255, 0);
    bp_sure_fg.convertTo(bp_sure_fg, CV_8U);

    imshow("Foreground image for red pills", rp_sure_fg);
    imshow("Foreground image for blue pills", bp_sure_fg);

    // Background
    Mat rp_sure_bg, bp_sure_bg;
    dilate(rp_th, rp_sure_bg, kernel, Point(-1,-1), 3);
    dilate(bp_th, bp_sure_bg, kernel, Point(-1,-1), 3);

    imshow("Background image for red pills", rp_sure_bg);
    imshow("Background image for blue pills", bp_sure_bg);

    // Unknown region
    Mat rp_unknown, bp_unknown;
    subtract(rp_sure_bg, rp_sure_fg, rp_unknown);
    subtract(bp_sure_bg, bp_sure_fg, bp_unknown);

    imshow("Unknown region for red pills", rp_unknown);
    imshow("Unknown region for blue pills", bp_unknown);

    // Connected components on sure_fg
    Mat rp_markers, bp_markers;
    connectedComponents(rp_sure_fg, rp_markers);
    connectedComponents(bp_sure_fg, bp_markers);

    // Shift labels so background = 1
    rp_markers = rp_markers + 1;
    bp_markers = bp_markers + 1;

    // Unknown region = 0
    rp_markers.setTo(0, rp_unknown == 255);
    bp_markers.setTo(0, bp_unknown == 255);

    // Apply watershed
    watershed(red_pill_img, rp_markers);
    watershed(blue_pill_img, bp_markers);

    // Re-map regions & count
    int num_red_pills = countObjects(rp_markers);
    int num_blue_pills = countObjects(bp_markers);
    
    cout << "Number of red pills detected: " << num_red_pills << endl;
    cout << "Number of blue pills detected: " << num_blue_pills << endl;
    // Optional: Draw boundaries
    Mat rp_result = red_pill_img.clone();
    rp_result.setTo(Scalar(0,0,255), rp_markers == -1); // boundaries in red
    Mat bp_result = blue_pill_img.clone();
    bp_result.setTo(Scalar(0,0,255), bp_markers == -1); // boundaries in red

    imshow("Result for red pills", rp_result);
    imshow("Result for blue pills", bp_result);
    
    waitKey(0);
    return 0;
}