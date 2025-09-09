#include <opencv2/opencv.hpp>
#include <algorithm>
#include "pill_counter.hpp"

using namespace cv;
using namespace std;

namespace pilseg {

    // ---------- Helpers: Luminance (L*), CLAHE, and thresholding ----------

    double autoClipLimit(const cv::Mat& L8, double clipMin, double clipMax) {
        CV_Assert(L8.type() == CV_8U);
        cv::Scalar mean, stddev; cv::meanStdDev(L8, mean, stddev);
        double t = std::min(1.0, std::max(0.0, stddev[0] / 64.0));
        return clipMin + t * (clipMax - clipMin);
    }

    int autoBlockSize(const cv::Size& sz) {
        int base = std::max(15, (std::min(sz.width, sz.height) / 16) * 2 + 1);
        return (base % 2 == 0) ? base + 1 : base;
    }

    // Build luminance mask (0/255) that downplays soft shadows.
    // mode: "adaptive" (shadow-robust) or "otsu" (global)
    cv::Mat luminanceMask(const cv::Mat& bgr, const std::string& mode, cv::Size tileGrid)
    {
        CV_Assert(!bgr.empty() && bgr.channels() == 3);
        cv::Mat lab8; cv::cvtColor(bgr, lab8, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> ch; cv::split(lab8, ch);
        cv::Mat L8 = ch[0];

        double clip = autoClipLimit(L8);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip, tileGrid);
        cv::Mat Leq; clahe->apply(L8, Leq);
        cv::GaussianBlur(Leq, Leq, cv::Size(3,3), 0);

        cv::Mat mask;
        if (mode == "otsu") {
            cv::threshold(Leq, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        } else {
            int block = autoBlockSize(Leq.size());
            cv::Scalar m, s; cv::meanStdDev(Leq, m, s);
            double C = std::max(2.0, std::min(12.0, 0.05 * s[0]));
            cv::adaptiveThreshold(Leq, mask, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv::THRESH_BINARY, block, C);
        }
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
        return mask;
    }

    // ---------- Helpers: Chroma magnitude from Lab (a*, b*) ----------

    // Return C = sqrt(a^2 + b^2) as CV_32F, computed on color-stabilized Lab (float)
    cv::Mat chromaMagnitudeF32(const cv::Mat& bgr) {
       CV_Assert(!bgr.empty() && bgr.channels() == 3);
        cv::Mat f32; bgr.convertTo(f32, CV_32F, 1.0/255.0);
        cv::Mat lab32f; cv::cvtColor(f32, lab32f, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> lab; cv::split(lab32f, lab);
        cv::Mat a = lab[1], b = lab[2];

        cv::Mat a2, b2, C;
        cv::multiply(a, a, a2);
        cv::multiply(b, b, b2);
        cv::add(a2, b2, C);
        cv::sqrt(C, C);
        return C; // CV_32F
    }

    // Normalize float chroma to 8-bit for global thresholding/viz
    cv::Mat normalizeToU8(const Mat& src32f) {
        double minv, maxv; cv::minMaxLoc(src32f, &minv, &maxv);
        if (maxv <= minv) return cv::Mat::zeros(src32f.size(), CV_8U);
        cv::Mat scaled, out = (src32f - minv) * (255.0 / (maxv - minv));
        scaled = out; // alias
        scaled.convertTo(out, CV_8U);
        return out;
    }

    //function for creation of chroma mask
    cv::Mat chromaMask(const cv::Mat& bgr,
                const std::string& mode) // "otsu" | "kmeans"
    {
        cv::Mat C32 = chromaMagnitudeF32(bgr);

        if (mode == "kmeans") {
            cv::Mat samples = C32.reshape(1, (int)C32.total());
            cv::Mat labels, centers;
            cv::kmeans(samples, 2, labels,
                    cv::TermCriteria(cv::TermCriteria::EPS|cv::TermCriteria::COUNT, 20, 1e-3),
                    2, cv::KMEANS_PP_CENTERS, centers);
            int fgLab = (centers.at<float>(0) > centers.at<float>(1)) ? 0 : 1;

            cv::Mat mask = cv::Mat::zeros(C32.size(), CV_8U);
            cv::Mat lblImg = labels.reshape(1, C32.rows);
            mask.setTo(255, (lblImg == fgLab));
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
            return mask;
        } else {
            cv::Mat C8 = normalizeToU8(C32), mask;
            cv::threshold(C8, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
            return mask;
        }
    }

    // Build fused mask = luminanceMask ∧ chromaMask
    FusionOutputs fusedMaskForWatershed(const cv::Mat& bgr,
                                        const std::string& lumMode,
                                        const std::string& chromaMode)
    {
        FusionOutputs out;

        out.lumMask    = luminanceMask(bgr, lumMode);
        out.chromaMask = chromaMask(bgr, chromaMode);
        cv::bitwise_and(out.lumMask, out.chromaMask, out.fused);
        cv::morphologyEx(out.fused, out.fused, cv::MORPH_CLOSE,
                        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
        return out;
    }

    cv::Mat makeSeedsFromFused(const cv::Mat& fused, double fgPercentile) {
        CV_Assert(fused.type() == CV_8U);
        cv::Mat bin; cv::threshold(fused, bin, 0, 255, cv::THRESH_BINARY);
        cv::Mat dt; cv::distanceTransform(bin, dt, cv::DIST_L2, 5);

        std::vector<float> v; v.reserve((size_t)dt.total());
        for (int r = 0; r < dt.rows; ++r) {
            const float* p = dt.ptr<float>(r);
            const uchar* m = bin.ptr<uchar>(r);
            for (int c = 0; c < dt.cols; ++c) if (m[c]) v.push_back(p[c]);
        }
        float thr = 0.f;
        if (!v.empty()) {
            size_t k = (size_t)std::round(fgPercentile * (v.size() - 1));
            std::nth_element(v.begin(), v.begin()+k, v.end());
            thr = v[k];
        }
        cv::Mat sureFg; cv::threshold(dt, sureFg, thr, 255, cv::THRESH_BINARY); sureFg.convertTo(sureFg, CV_8U);
        cv::Mat sureBg; cv::dilate(bin, sureBg, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)), cv::Point(-1,-1), 2);
        cv::Mat unknown; cv::subtract(sureBg, sureFg, unknown);

        cv::Mat markers; cv::connectedComponents(sureFg, markers);
        markers += 1;
        markers.setTo(0, unknown > 0);
        return markers;
    }

    WSOut runWatershed(const cv::Mat& bgr, const cv::Mat& fused, double fgPercentile) {
        CV_Assert(!bgr.empty() && bgr.channels() == 3 && fused.type() == CV_8U);
        cv::Mat markers = makeSeedsFromFused(fused, fgPercentile);

        cv::Mat img = bgr.clone();
        cv::watershed(img, markers);

        cv::Mat segMask = cv::Mat::zeros(fused.size(), CV_8U);
        segMask.setTo(255, markers > 1);
        cv::morphologyEx(segMask, segMask, cv::MORPH_CLOSE,
                        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));

        cv::Mat labels, stats, cents;
        int n = cv::connectedComponentsWithStats(segMask, labels, stats, cents, 8, CV_32S);

        int areaMin = std::max(64, (segMask.rows * segMask.cols) / 10000);
        cv::Mat clean = cv::Mat::zeros(segMask.size(), CV_8U);
        std::vector<cv::Rect> boxes;
        for (int i = 1; i < n; ++i) {
            int a = stats.at<int>(i, cv::CC_STAT_AREA);
            if (a >= areaMin) {
                clean.setTo(255, labels == i);
                boxes.emplace_back(
                    stats.at<int>(i, cv::CC_STAT_LEFT),
                    stats.at<int>(i, cv::CC_STAT_TOP),
                    stats.at<int>(i, cv::CC_STAT_WIDTH),
                    stats.at<int>(i, cv::CC_STAT_HEIGHT)
                );
            }
        }
        return WSOut{clean, markers, boxes};
    }
} //namespace: pilseg