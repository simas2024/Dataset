//
//  DatasetGenerator.cpp
//  Dataset
//
//  Created by Markus Schmid on 27.03.25.
//

#include "DatasetGenerator.hpp"
#include <thread>

using namespace cv;

namespace {
std::vector<cv::Rect> boundingBoxesAusKonturen(const std::vector<std::vector<cv::Point>> &konturen) {
    std::vector<cv::Rect> boxen;
    for (const auto &kontur : konturen) {
        boxen.push_back(cv::boundingRect(kontur));
    }
    return boxen;
}

cv::Point2f berechneSchwerpunkt(const std::vector<cv::Point> &kontur) {
    cv::Moments m = cv::moments(kontur);
    if (m.m00 != 0)
        return cv::Point2f(static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00));
    else
        return cv::Point2f(0, 0); // Leere oder ungültige Kontur
}

cv::Mat kontrast(const cv::Mat &gray_image, const ErkennungKonfig &config) {
    cv::Mat result;
    cv::convertScaleAbs(gray_image, result, config.kontrast_alpha, config.kontrast_beta);
    return result;
}

std::string formatTime(double seconds) {
    int h = static_cast<int>(seconds / 3600);
    int m = static_cast<int>((seconds - h * 3600) / 60);
    int s = static_cast<int>(seconds) % 60;
    char buffer[9];
    snprintf(buffer, sizeof(buffer), "%02d:%02d:%02d", h, m, s);
    return string(buffer);
}

void drawConfig(cv::Mat &img, const ErkennungKonfig &config, const int referenzFrameNummer) {
    int y = 20;
    const int lineHeight = 20;
    const cv::Scalar color(0, 255, 0); // Grün
    const double fontScale = 0.5;
    const int thickness = 1;
    const int fontFace = cv::FONT_HERSHEY_SIMPLEX;

    auto drawLine = [&](const std::string &text) {
        cv::putText(img, text, cv::Point(10, y), fontFace, fontScale, color, thickness);
        y += lineHeight;
    };

    drawLine("Referenz Frame: " + std::to_string(referenzFrameNummer));
    drawLine("kontrast_alpha: " + std::to_string(config.kontrast_alpha));
    drawLine("kontrast_beta: " + std::to_string(config.kontrast_beta));
    drawLine("gauss_deviation: " + std::to_string(config.gauss_deviation));
    drawLine("boundingbox_min_ratio: " + std::to_string(config.boundingbox_min_ratio));
    drawLine("boundingbox_min_area: " + std::to_string(config.boundingbox_min_area));
    drawLine("boundingbox_max_area: " + std::to_string(config.boundingbox_max_area));
    drawLine("thresh_thresh: " + std::to_string(config.thresh_thresh));
    drawLine("thresh_maxval: " + std::to_string(config.thresh_maxval));
    drawLine("thres_type: " + std::to_string(config.thres_type));
    drawLine("find_contours_mode: " + std::to_string(config.find_contours_mode));
    drawLine("find_contours_method: " + std::to_string(config.find_contours_method));
}

} // namespace

namespace masc::annotations {
std::map<int, cv::Mat> DatasetGenerator::referenzFramePuffer;

int DatasetGenerator::referenzFrameFuerIndex(const std::string &jsonPfad, int frameIndex) {
    std::ifstream file(jsonPfad);
    if (!file) {
        return -1;
    }

    nlohmann::json j;
    file >> j;

    for (const auto &segment : j["segments"]) {
        int start = segment.at("start");
        int end = segment.at("end");
        int ref = segment.at("ref");

        if (frameIndex >= start && frameIndex <= end) {
            return ref;
        }
    }
    return -1;
}

bool DatasetGenerator::istUnscharf(const cv::Mat &ausschnitt, double schwellenwert) {
    cv::Mat laplace, gray;
    if (ausschnitt.channels() == 3)
        cv::cvtColor(ausschnitt, gray, cv::COLOR_BGR2GRAY);
    else
        gray = ausschnitt;

    cv::Laplacian(gray, laplace, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplace, mean, stddev);
    double varianz = stddev[0] * stddev[0];

    return varianz < schwellenwert;
}

cv::Mat DatasetGenerator::ladeReferenzFrame(const std::string &videopfad, int frame_index, const cv::Mat &maske, const ErkennungKonfig &config) {
    cv::VideoCapture cap(videopfad);
    if (!cap.isOpened()) {
        throw std::runtime_error("❌ Fehler beim Öffnen des Videos: " + videopfad);
    }

    cap.set(cv::CAP_PROP_POS_FRAMES, frame_index);
    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) {
        throw std::runtime_error("❌ Referenzframe leer bei Index " + std::to_string(frame_index));
    }
    
    // Graustufen
    cv::Mat frame_gray, frame_gray_blurred;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

    if (!maske.empty()) {
        cv::Mat frame_gray_blurred_kontrast_masked;
        GaussianBlur(frame_gray, frame_gray_blurred, Size(config.gauss_deviation, config.gauss_deviation), 0);
        cv::Mat frame_gray_blurred_kontrast = kontrast(frame_gray_blurred, config);
        cv::bitwise_and(frame_gray_blurred_kontrast, frame_gray_blurred_kontrast, frame_gray_blurred_kontrast_masked, maske);

        if (config.debug > 0) {
            cv::imshow("maske", maske);
            cv::waitKey(0); // wartet auf Tastendruck
            cv::imshow("frame_gray", frame_gray);
            cv::waitKey(0); // wartet auf Tastendruck
            cv::imshow("frame_gray_blurred", frame_gray_blurred);
            cv::waitKey(0); // wartet auf Tastendruck
            cv::imshow("frame_gray_blurred_kontrast", frame_gray_blurred_kontrast);
            cv::waitKey(0); // wartet auf Tastendruck
            cv::imshow("frame_gray_blurred_kontrast_masked", frame_gray_blurred_kontrast_masked);
            cv::waitKey(0); // wartet auf Tastendruck
            cv::destroyAllWindows();
        }

        return frame_gray_blurred_kontrast_masked;
    }

    return frame_gray;
}

std::vector<std::tuple<
    std::vector<std::vector<cv::Point>>, // konturen
    std::vector<cv::Rect>,               // boxen
    std::vector<cv::Rect>                // gefilterte_boxen
>> DatasetGenerator::boundingBoxes(const std::vector<cv::Mat>& frames, const cv::Mat &maske_sw, const cv::Mat &ref_frame_gray,
                                const ErkennungKonfig &config) {
    std::vector<std::thread> threads;

    std::vector<cv::Mat> frame_aktuell_gray(frames.size());
    std::vector<cv::Mat> frame_aktuell_gray_blurred(frames.size());
    std::vector<cv::Mat> frame_aktuell_gray_blurred_kontrast(frames.size());
    std::vector<cv::Mat> frame_aktuell_gray_blurred_kontrast_masked(frames.size());

    using Resultat = std::tuple<
        std::vector<std::vector<cv::Point>>, // konturen
        std::vector<cv::Rect>,               // boxen
        std::vector<cv::Rect>                // gefilterte_boxen
    >;

    std::vector<Resultat> resultate(frames.size());
    
    for (int i = 0; i < (int)frames.size(); ++i) {
        threads.emplace_back([&, i]() {
            cvtColor(frames[i], frame_aktuell_gray[i], cv::COLOR_BGR2GRAY);
            cv::Size ksize(config.gauss_deviation, config.gauss_deviation);
            GaussianBlur(frame_aktuell_gray[i], frame_aktuell_gray_blurred[i], ksize, 0);
            frame_aktuell_gray_blurred_kontrast[i] = kontrast(frame_aktuell_gray_blurred[i], config);
            bitwise_and(
                frame_aktuell_gray_blurred_kontrast[i],
                frame_aktuell_gray_blurred_kontrast[i],
                frame_aktuell_gray_blurred_kontrast_masked[i],
                maske_sw
            );
            
            Mat diff, thresh;
            
            if (ref_frame_gray.empty() || frame_aktuell_gray_blurred_kontrast_masked[i].empty()) {
                std::cerr << "Einer der Inputs ist leer!\n";
            } else if (ref_frame_gray.size() != frame_aktuell_gray_blurred_kontrast_masked[i].size() || ref_frame_gray.type() != frame_aktuell_gray_blurred_kontrast_masked[i].type()) {
                std::cerr << "Inputs haben unterschiedliche Größe oder Typ!\n";
            } else {
                cv::absdiff(ref_frame_gray, frame_aktuell_gray_blurred_kontrast_masked[i], diff);
            }
            
            threshold(diff, thresh, config.thresh_thresh, config.thresh_maxval, config.thres_type);
            vector<vector<Point>> konturen;
            findContours(thresh, konturen, config.find_contours_mode, config.find_contours_method);

            std::vector<cv::Rect> boxen;
            for (const auto& k : konturen) {
                boxen.push_back(boundingRect(k));
            }

            std::vector<cv::Rect> gefilterte_boxen;

            for (const auto &box : boxen) {
                if (((float)std::min(box.width, box.height) / std::max(box.width, box.height) > config.boundingbox_min_ratio) &&
                    box.area() > config.boundingbox_min_area && box.area() < config.boundingbox_max_area) {
                    cv::Mat ausschnitt = frame_aktuell_gray[i](box);
                    if (!istUnscharf(ausschnitt, 50)) {
                        gefilterte_boxen.push_back(box);
                    }
                }
            }

            // ⬇️ Ergebnis in gemeinsamen Vektor schreiben
            resultate[i] = std::make_tuple(konturen, boxen, gefilterte_boxen);
        });
    }

    for (auto& t : threads) t.join();
    
    
    /*
        if (config.debug > 0) {
            cv::imshow("frame_aktuell", frame_aktuell);
            cv::waitKey(0); // wartet auf Tastendruck
            cv::imshow("frame_aktuell_gray", frame_aktuell_gray);
            cv::waitKey(0); // wartet auf Tastendruck
            cv::imshow("frame_aktuell_gray_blurred", frame_aktuell_gray_blurred);
            cv::waitKey(0); // wartet auf Tastendruck
            cv::imshow("frame_aktuell_gray_blurred_kontrast", frame_aktuell_gray_blurred_kontrast);
            cv::waitKey(0); // wartet auf Tastendruck
            cv::imshow("frame_aktuell_gray_blurred_kontrast_masked", frame_aktuell_gray_blurred_kontrast_masked);
            cv::waitKey(0); // wartet auf Tastendruck
            cv::imshow("ref_frame_gray", ref_frame_gray);
            cv::waitKey(0); // wartet auf Tastendruck
            cv::destroyAllWindows();
        }
     */
    return resultate;
};

std::tuple<std::vector<std::vector<cv::Point>>, std::vector<cv::Rect>, std::vector<cv::Rect>, cv::Mat>
DatasetGenerator::boundingBoxes(cv::VideoCapture &cap, int frame_index, const cv::Mat &maske_sw, const cv::Mat &ref_frame_gray,
                                const ErkennungKonfig &config) {
    std::vector<cv::Rect> boxen;

    if (!cap.isOpened()) {
        cerr << "❌ Fehler beim Öffnen des Videos!" << endl;
        return {};
    }

    cap.set(CAP_PROP_POS_FRAMES, frame_index);

    Mat frame_aktuell;
    cap >> frame_aktuell;
    if (frame_aktuell.empty()) {
        cerr << "❌ Aktueller Frame " << frame_index << " ist leer!" << endl;
        return {};
    }

    if (frame_aktuell.size() != maske_sw.size()) {
        cerr << "❌ Maske passt nicht zur Frame-Größe!" << endl;
        return {};
    }

    cv::Mat frame_aktuell_gray, frame_aktuell_gray_blurred, frame_aktuell_gray_blurred_kontrast, frame_aktuell_gray_blurred_kontrast_masked;

    cvtColor(frame_aktuell, frame_aktuell_gray, COLOR_BGR2GRAY);

    GaussianBlur(frame_aktuell_gray, frame_aktuell_gray_blurred, Size(config.gauss_deviation, config.gauss_deviation), 0);

    frame_aktuell_gray_blurred_kontrast = kontrast(frame_aktuell_gray_blurred, config);

    bitwise_and(frame_aktuell_gray_blurred_kontrast, frame_aktuell_gray_blurred_kontrast, frame_aktuell_gray_blurred_kontrast_masked, maske_sw);

    if (config.debug > 0) {
        cv::imshow("frame_aktuell", frame_aktuell);
        cv::waitKey(0); // wartet auf Tastendruck
        cv::imshow("frame_aktuell_gray", frame_aktuell_gray);
        cv::waitKey(0); // wartet auf Tastendruck
        cv::imshow("frame_aktuell_gray_blurred", frame_aktuell_gray_blurred);
        cv::waitKey(0); // wartet auf Tastendruck
        cv::imshow("frame_aktuell_gray_blurred_kontrast", frame_aktuell_gray_blurred_kontrast);
        cv::waitKey(0); // wartet auf Tastendruck
        cv::imshow("frame_aktuell_gray_blurred_kontrast_masked", frame_aktuell_gray_blurred_kontrast_masked);
        cv::waitKey(0); // wartet auf Tastendruck
        cv::imshow("ref_frame_gray", ref_frame_gray);
        cv::waitKey(0); // wartet auf Tastendruck
        cv::destroyAllWindows();
    }

    Mat diff, thresh;
    absdiff(ref_frame_gray, frame_aktuell_gray_blurred_kontrast_masked, diff);
    threshold(diff, thresh, config.thresh_thresh, config.thresh_maxval, config.thres_type);

    vector<vector<Point>> konturen;
    findContours(thresh, konturen, config.find_contours_mode, config.find_contours_method);

    boxen = boundingBoxesAusKonturen(konturen);

    std::vector<cv::Rect> gefilterte_boxen;

    Mat frame_mit_box = frame_aktuell.clone();
    for (const auto &box : boxen) {
        // cout << "box.area() = " << box.area() << endl;

        if ((((float)std::min(box.width, box.height) / std::max(box.width, box.height) > config.boundingbox_min_ratio)) &&
            box.area() > config.boundingbox_min_area && box.area() < config.boundingbox_max_area) {
            cv::Mat ausschnitt = frame_aktuell_gray(box);
            if (!istUnscharf(ausschnitt, 30)) {
                gefilterte_boxen.push_back(box);
            }
        }
    }

    return std::make_tuple(konturen, boxen, gefilterte_boxen, frame_aktuell);
};

void DatasetGenerator::speichereFuerYolo(const cv::Mat& frame,
                                         const std::vector<cv::Rect>& boxen,
                                         const std::string& ausgabePfad,
                                         const std::string& basename,
                                         int class_id) {
    // Speicherpfade
    std::string imagePath = ausgabePfad + basename + ".jpg";
    std::string labelPath = ausgabePfad + basename + ".txt";

    // Bild speichern
    cv::imwrite(imagePath, frame);

    // Textdatei für Labels öffnen
    std::ofstream out(labelPath);
    if (!out.is_open()) {
        std::cerr << "❌ Konnte Datei nicht schreiben: " << labelPath << std::endl;
        return;
    }

    int img_width = frame.cols;
    int img_height = frame.rows;

    for (const auto& box : boxen) {
        float x_center = (box.x + box.width / 2.0f) / img_width;
        float y_center = (box.y + box.height / 2.0f) / img_height;
        float width = static_cast<float>(box.width) / img_width;
        float height = static_cast<float>(box.height) / img_height;

        out << class_id << " "
            << x_center << " " << y_center << " "
            << width << " " << height << std::endl;
    }

    out.close();
}

void DatasetGenerator::speichereFuerCreateML(const cv::Mat& frame,
                                             const std::vector<cv::Rect>& boxen,
                                             const std::string& ausgabePfad,
                                             const std::string& basename,
                                             const std::string& label_name) {
    // Speicherpfade
    std::string imagePath = ausgabePfad + "images/" + basename + ".jpg";
    std::string jsonPath  = ausgabePfad + "annotations/" + basename + ".json";
    
    // Zielordner erzeugen, falls nötig
    std::filesystem::create_directories(ausgabePfad + "images/");
    std::filesystem::create_directories(ausgabePfad + "annotations/");

    // Bild speichern
    cv::imwrite(imagePath, frame);

    // JSON-Objekt aufbauen
    json j;
    j["image"] = basename + ".jpg";
    j["annotations"] = json::array();

    for (const auto& box : boxen) {
        float x_center = box.x + box.width / 2.0f;
        float y_center = box.y + box.height / 2.0f;

        json eintrag;
        eintrag["label"] = label_name;
        eintrag["coordinates"] = {
            {"x", x_center},
            {"y", y_center},
            {"width", box.width},
            {"height", box.height}
        };
        j["annotations"].push_back(eintrag);
    }

    // JSON-Datei speichern
    std::ofstream out(jsonPath);
    if (!out.is_open()) {
        std::cerr << "❌ Konnte Datei nicht schreiben: " << jsonPath << std::endl;
        return;
    }

    out << std::setw(2) << j << std::endl;
    out.close();
}

void DatasetGenerator::debugDataset(const std::vector<std::vector<cv::Point>> &konturen, const std::vector<cv::Rect> &boxen,
                                      const std::vector<cv::Rect> &filteredBoxen, const cv::Mat &referenzFrame, const cv::Mat &frame,
                                      const int num, const int numReferenzFrame, const std::string &ausgabePfad, const ErkennungKonfig &config) {
    cv::Mat frameMitBoxen = frame.clone();
    cv::Mat frameMitFilteredBoxen = frame.clone();
    cv::Mat frameMitKonturen = frame.clone();

    // filteredBoxen
    for (const auto &box : boxen) {
        cv::rectangle(frameMitBoxen, box, cv::Scalar(0, 255, 0), 1); // grün
    }

    // filtereframeMitFilteredBoxendBoxen
    for (const auto &box : filteredBoxen) {
        cv::rectangle(frameMitFilteredBoxen, box, cv::Scalar(0, 255, 0), 1); // grün
    }

    // frameMitKonturen
    cv::drawContours(frameMitKonturen, konturen, -1, cv::Scalar(0, 255, 0), 1);

    cv::Mat referenzFrame_bgr;
    if (referenzFrame.channels() == 1)
        cv::cvtColor(referenzFrame, referenzFrame_bgr, cv::COLOR_GRAY2BGR);
    else
        referenzFrame_bgr = referenzFrame.clone();

    int zielHoehe = referenzFrame.rows;

    auto ensure3ChannelsAndResize = [zielHoehe](cv::Mat &img) {
        if (img.channels() == 1) {
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        }
        if (img.rows != zielHoehe) {
            double scale = static_cast<double>(zielHoehe) / img.rows;
            cv::resize(img, img, cv::Size(), scale, scale);
        }
    };

    ensure3ChannelsAndResize(referenzFrame_bgr);
    ensure3ChannelsAndResize(frameMitKonturen);
    ensure3ChannelsAndResize(frameMitBoxen);
    ensure3ChannelsAndResize(frameMitFilteredBoxen);

    drawConfig(referenzFrame_bgr, config, numReferenzFrame);

    // Bilder nebeneinander setzen
    cv::Mat nebeneinander;
    cv::hconcat(std::vector<cv::Mat>{referenzFrame_bgr, frameMitKonturen, frameMitBoxen, frameMitFilteredBoxen}, nebeneinander);

    std::ostringstream oss;
    std::string dateiname = std::format("/frame_{:06}", num);
    oss << ausgabePfad << dateiname << ".png";

    cv::imwrite(oss.str(), nebeneinander);
}

void DatasetGenerator::initReferenzFrames(const std::string &jsonPfad, const std::string &videopfad, const cv::Mat &maske,
                                          const ErkennungKonfig &config) {
    std::ifstream in(jsonPfad);
    if (!in.is_open()) {
        std::cerr << "❌ Fehler: JSON-Datei konnte nicht geöffnet werden: " << jsonPfad << std::endl;
        return;
    }

    json ref_frame_config;
    in >> ref_frame_config;

    std::set<int> referenzen;

    for (const auto &segment : ref_frame_config["segments"]) {
        if (segment.contains("ref")) {
            referenzen.insert(segment["ref"].get<int>());
        }
    }

    for (int refIndex : referenzen) {
        if (referenzFramePuffer.count(refIndex) == 0) {
            cv::Mat frame = ladeReferenzFrame(videopfad, refIndex, maske, config);
            referenzFramePuffer[refIndex] = frame;
        }
    }
}

const cv::Mat &DatasetGenerator::getReferenzFrame(int refIndex) {
    return referenzFramePuffer.at(refIndex); // throws wenn nicht vorhanden
}

} // Namespace masc::annotations
