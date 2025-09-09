#include "pill_counter.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat p1_img = cv::imread("images/blue-pills-white-bg.jpg");
    cv::Mat p2_img = cv::imread("images/red-pills-white-bg.jpg");

    // for blue pills
    auto p1_out = pilseg::fusedMaskForWatershed(p1_img, "adaptive", "kmeans");
    auto p1_ws   = pilseg::runWatershed(p1_img, p1_out.fused, 0.65);
    for (auto& r : p1_ws.boxes) rectangle(p1_img, r, Scalar(0,255,0), 2);
    cv::imshow("Boxes for blue pills", p1_img);
    int p1_cnt = static_cast<int>(p1_ws.boxes.size());
    cout << "Blue pills count: " << p1_cnt << "\n";
    cv::imwrite("results/boxed_blue_pills.jpg", p1_img);

    //for red pills
    auto p2_out = pilseg::fusedMaskForWatershed(p2_img, "adaptive", "kmeans");
    auto p2_ws   = pilseg::runWatershed(p2_img, p2_out.fused, 0.65);
    for (auto& r : p2_ws.boxes) rectangle(p2_img, r, Scalar(0,255,0), 2);
    cv::imshow("Boxes for red pills", p2_img);
    int p2_cnt = static_cast<int>(p2_ws.boxes.size());
    cout << "Red pills count: " << p2_cnt << "\n";
    cv::imwrite("results/boxed_red_pills.jpg", p2_img);

    // cv::imshow("Lum Mask", outs.lumMask);
    // cv::imshow("Chroma Mask", outs.chromaMask);
    // cv::imshow("Fused", outs.fused);
    // cv::imshow("Segments", ws.segMask);

    cv::waitKey(0);
    return 0;
}
