// Which arithmetic does cv::GaussianBlur(CV_64F column, Size(1, 0), sigma) use on this build?
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstdio>
int main() {
    const int n = 300, r = 16;
    cv::Mat col(n, 1, CV_64F), bl;
    for (int i = 0; i < n; ++i) col.at<double>(i) = 100 + 37.3 * std::sin(i * 0.1) + i * 0.731 + (i % 7) * 0.013;
    cv::GaussianBlur(col, bl, cv::Size(1, 0), 4);
    cv::Mat kk = cv::getGaussianKernel(2 * r + 1, 4, CV_64F);
    const double* k = kk.ptr<double>();
    auto X = [&](int i) { if (i < 0) i = -i; if (i >= n) i = 2 * n - 2 - i; return col.at<double>(i); };   // REFLECT_101
    int bad[4] = {0, 0, 0, 0};
    for (int i = 0; i < n; ++i) {
        double a = k[r] * X(i), b = k[r] * X(i), c = 0, d = 0;
        for (int j = 1; j <= r; ++j) { a += k[r + j] * (X(i - j) + X(i + j)); b = std::fma(k[r + j], X(i - j) + X(i + j), b); }
        for (int j = 0; j <= 2 * r; ++j) { c += k[j] * X(i - r + j); d = std::fma(k[j], X(i - r + j), d); }
        double o = bl.at<double>(i);
        bad[0] += a != o; bad[1] += b != o; bad[2] += c != o; bad[3] += d != o;
    }
    std::printf("mismatches of %d: symm %d, symm+fma %d, seq %d, seq+fma %d\n", n, bad[0], bad[1], bad[2], bad[3]);
    std::printf("%s\n", cv::getBuildInformation().find("FMA") != std::string::npos ? "build mentions FMA" : "no FMA in build info");
    std::printf("checkHardwareSupport AVX2 %d FMA3 %d\n", (int)cv::checkHardwareSupport(CV_CPU_AVX2), (int)cv::checkHardwareSupport(CV_CPU_FMA3));
}
