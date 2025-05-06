//
//  DatasetGenerator.hpp
//  Dataset
//
//  Created by Markus Schmid on 27.03.25.
//


#ifndef DATASETGENERATOR_HPP
#define DATASETGENERATOR_HPP

#include <chrono>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace cv;
using namespace std;
using namespace nlohmann;

struct ErkennungKonfig {
    int debug = 0;
    double kontrast_alpha = 1.0;
    double kontrast_beta = 0.0;
    int gauss_deviation = 0;
    float boundingbox_min_ratio = 0.45f;
    int boundingbox_min_area = 500;
    int boundingbox_max_area = 10000;
    double thresh_thresh = 15.0;
    double thresh_maxval = 255.0;
    int thres_type = THRESH_BINARY & THRESH_OTSU;
    int find_contours_mode = RETR_EXTERNAL;
    int find_contours_method = CHAIN_APPROX_TC89_KCOS;
    int plausibilität_max_anzahl_boxen = 8;
    int plausibilität_max_bewegungsframes = 900;
};

namespace masc::annotations {

class DatasetGenerator {

  public:
    cv::Mat ladeReferenzFrame(const std::string &videopfad, int frame_index, const cv::Mat &maske = cv::Mat(),
                              const ErkennungKonfig &config = ErkennungKonfig());

    std::tuple<std::vector<std::vector<cv::Point>>, std::vector<cv::Rect>, std::vector<cv::Rect>, cv::Mat>
    boundingBoxes(cv::VideoCapture &cap, int frame_index, const cv::Mat &maske_sw, const cv::Mat &ref_frame_maskiert_gray,
                  const ErkennungKonfig &config = ErkennungKonfig());

    std::vector<std::tuple<
        std::vector<std::vector<cv::Point>>, // konturen
        std::vector<cv::Rect>,               // boxen
        std::vector<cv::Rect>                // gefilterte_boxen
    >>
    boundingBoxes(const std::vector<cv::Mat>& frames, const cv::Mat &maske_sw, const cv::Mat &ref_frame_gray,
                                    const ErkennungKonfig &config);
    
    void debugDataset(const std::vector<std::vector<cv::Point>> &konturen, const std::vector<cv::Rect> &boxen,
                        const std::vector<cv::Rect> &filteredBoxen, const cv::Mat &referenzFrame, const cv::Mat &frame, const int num, const int numReferenzFrame,
                        const std::string &ausgabePfad, const ErkennungKonfig &config = ErkennungKonfig());

    void speichereFuerYolo(const cv::Mat& frame,
                                             const std::vector<cv::Rect>& boxen,
                                             const std::string& ausgabePfad,
                                             const std::string& basename,
                                             int class_id = 0);
    void speichereFuerCreateML(const cv::Mat& frame,
                                                 const std::vector<cv::Rect>& boxen,
                                                 const std::string& ausgabePfad,
                                                 const std::string& basename,
                               const std::string& label_name);
    
    void initReferenzFrames(const std::string &jsonPfad, const std::string &videopfad, const cv::Mat &maske,
                            const ErkennungKonfig &config = ErkennungKonfig());

    static const cv::Mat &getReferenzFrame(int refIndex);

    int referenzFrameFuerIndex(const std::string &jsonPfad, int frameIndex);

  private:
    // Interne Hilfsmethoden
    std::string formatTime(double seconds);
    bool istUnscharf(const cv::Mat &ausschnitt, double schwellenwert = 100.0);
    static std::map<int, cv::Mat> referenzFramePuffer;
};

} // Namespace masc::annotations

#endif // DATASETGENERATOR_HPP
