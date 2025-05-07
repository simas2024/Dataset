//
//  main.m
//  Dataset
//
//  Created by Markus Schmid on 27.03.25.
//

#include "DatasetGenerator.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace cv;
using namespace std;
using namespace masc::annotations;
namespace fs = std::filesystem;

DatasetGenerator gen;
ErkennungKonfig config;
cv::Mat maske_bw;
cv::VideoCapture cap;

std::string default_config_path;
std::string set_path;
std::string clip_path;
std::string json_path;
std::string maske_bw_path;
std::string debug_path;
std::string yolo_path;
std::string ml_path;
std::string clip_kennung;

constexpr const int CLASS_ID = 1;
constexpr const char* CLASS_NAME = "ant";

void setUp() {
    config.debug = 0;
    
    // config set 1
    /*
    config.gauss_deviation = 17;
    config.thresh_thresh = 16.0;
    config.boundingbox_min_area = 2000;
    config.boundingbox_max_area = 6000;
    config.boundingbox_min_ratio = 0.375;
    config.kontrast_alpha = 0.9;
    config.kontrast_beta = 0.0;
    */

    // config set 2
    config.gauss_deviation = 7;
    config.thresh_thresh = 15.0;
    config.boundingbox_min_area = 2000;
    config.boundingbox_max_area = 6000;
    config.boundingbox_min_ratio = 0.375;
    config.kontrast_alpha = 1.1;
    config.kontrast_beta = 0;
    
    default_config_path = std::string(PROJECT_SOURCE_DIR) + "/config/";
    // 20250324_1430 20250401_0945 20250314_1216
    clip_kennung = "20250415_1646";
    set_path = std::string(PROJECT_SOURCE_DIR) + "/data/" + clip_kennung + "/";
    clip_path = "/Volumes/HD16/clips/ants/clip_" + clip_kennung + ".mp4";
    maske_bw_path = set_path + "maske_sw.png";
    json_path = set_path + "clip.json";
    debug_path = set_path + "debug/";
    yolo_path = set_path + "yolo/";
    ml_path = set_path + "ml/";
    fs::create_directories(set_path);
    fs::create_directories(debug_path);
    fs::create_directories(yolo_path);
    fs::create_directories(ml_path);
    const std::vector<std::string> files = {"maske_sw.png", "clip.json"};
    for (const auto& file : files) {
        fs::path src = fs::path(default_config_path) / file;
        fs::path dst = fs::path(set_path) / file;

        if (!fs::exists(dst)) {
            try {
                fs::copy_file(src, dst, fs::copy_options::skip_existing);
            } catch (const fs::filesystem_error& e) {
                std::cerr << "Fehler beim Kopieren von " << file << ": " << e.what() << '\n';
            }
        }
    }
    cap.open(clip_path);
    maske_bw = cv::imread(maske_bw_path, cv::IMREAD_GRAYSCALE);
    gen.initReferenzFrames(json_path, clip_path, maske_bw, config);
};

int start_frame_nummer=0;
int end_frame_nummer=500000;

bool geringeErkennung(const std::vector<cv::Rect>& boxes, int min_count, int min_area_sum) {
    int area_sum = 0;
    for (const auto& box : boxes) {
        area_sum += box.area();
    }
    return (area_sum < min_area_sum);
}

int main() {
    
    struct FrameBlock {
        std::vector<cv::Mat> frames;
        int start_frame_index;
        std::vector<std::tuple<
            std::vector<std::vector<cv::Point>>, // konturen
            std::vector<cv::Rect>,               // boxen
            std::vector<cv::Rect>                // gefilterte_boxen
        >> ergebnisse;
        cv::Mat maske;
        cv::Mat ref_frame;
    };
    
    std::deque<FrameBlock> puffer_bewegung;
    
    const size_t max_puffer_size = 1000;
    
    setUp();

    if (!cap.isOpened()) {
        cerr << "❌ Fehler beim Öffnen des Videos!" << endl;
        return -1;
    }

    if (maske_bw.empty()) {
        cerr << "❌ Fehler beim Laden der Maske!" << endl;
        return -1;
    }
  
    int referenz_frame_num = 0;
    int neue_referenz_frame_num = referenz_frame_num;
    int frame_counter = start_frame_nummer; // 914 35456 35683 35780 26059 10447 148183 10423
   
    if (!cap.isOpened()) {
        cerr << "❌ Fehler beim Öffnen des Videos!" << endl;
        return {};
    }

    cap.set(CAP_PROP_POS_FRAMES, frame_counter);
  
    Mat ref_frame_maskiert_gray;
   
    bool objekt_erkannt = false;
   
    ostringstream filename;
    
    bool iterate=true;
    
    std::vector<cv::Rect> boxen;
   
    const int parallel = 20;
   
    std::vector<cv::Mat> frame_puffer;
    
    // Referenzframe setzen laut Konfiguration, falls vorhanden
    int refIdx = gen.referenzFrameFuerIndex(json_path, 0);
    if(refIdx>=0) {
        ref_frame_maskiert_gray = DatasetGenerator::getReferenzFrame(refIdx);
        referenz_frame_num = refIdx;
        neue_referenz_frame_num = refIdx;
    }
    
    while (iterate) {
        // std::cout << "Frame " << frame_counter << endl;
        iterate=true;

        // #parallel Frames setzen
        std::vector<cv::Mat> frames = frame_puffer;
        // Neue Frames ergänzen, bis frames.size() == parallel
        while (frames.size() < parallel) {
            cv::Mat frame_aktuell;
            cap >> frame_aktuell;
            if (frame_aktuell.empty()) break;
            frames.push_back(frame_aktuell);
        }
        
        if(frames.empty()) {
            break;
        }
        // Boundingboxen für Objekte suchen

        auto ergebnisse = gen.boundingBoxes(frames, maske_bw, ref_frame_maskiert_gray,config);
      
        int genutzte_ergebnisse=0;
        for (const auto& [gefundene_konturen, ungefilterte_boxen, gefilterte_boxen] : ergebnisse) {
            /* referenz_frames Kandidaten speichern:
             1. Falls schon lange kein neues Referenzframe mehr geladen wurde oder
             2. Falls innerhalb der letzten zehn Frames kein neues Referenzframe mehr geladen wurde und geringes Erkennungsmaß
             */
            if ((frame_counter+genutzte_ergebnisse>referenz_frame_num+1000) ||
                (geringeErkennung(ungefilterte_boxen,5,350) && frame_counter+genutzte_ergebnisse-referenz_frame_num>30)) {
                neue_referenz_frame_num = frame_counter+genutzte_ergebnisse;
            }
            
            // Auswerten, ob Objekte gefunden wurden
            objekt_erkannt=false;
            if (!gefilterte_boxen.empty()) {
                objekt_erkannt = true;
            }
            
            // Start der "Bewegung" erkennen
            if (objekt_erkannt && (puffer_bewegung.size()==0)) {
                puffer_bewegung.push_back(FrameBlock{frames, frame_counter+genutzte_ergebnisse, ergebnisse, maske_bw.clone(), ref_frame_maskiert_gray.clone()});
                std::cout << "🐜 zuerst bei Frame " << frame_counter+genutzte_ergebnisse << endl;
            }
            
            if (objekt_erkannt && (puffer_bewegung.size()>0)) {
                // Block in den Puffer
                // std::cout << "Schreibe Frame " << frame_counter+genutzte_ergebnisse << " in Bewegungsbuffer. #Boxen = " << gefilterte_boxen.size() << endl;
                if (puffer_bewegung.size() > 1) {
                    puffer_bewegung.push_back(FrameBlock{frames, frame_counter+genutzte_ergebnisse, ergebnisse, maske_bw.clone(), ref_frame_maskiert_gray.clone()});
                }
                if (puffer_bewegung.size() > max_puffer_size) {
                    puffer_bewegung.pop_front();
                }
                // std::cout << "🐜 Bewegung bei Frame " << frame_counter+genutzte_ergebnisse << endl;
            }
            
            // Ende der "Bewegung" erkennen
            if ((!objekt_erkannt && (puffer_bewegung.size()>0)) || (int)puffer_bewegung.size()>config.plausibilität_max_bewegungsframes) {
                std::cout << "🐜 zuletzt bei Frame " << frame_counter+genutzte_ergebnisse-1 << endl;
                // Auswertung
                size_t mitte = puffer_bewegung.size() / 2;
                const FrameBlock& datensatz = puffer_bewegung[mitte];
                int start_frame_index = datensatz.start_frame_index;
                std::vector<std::vector<cv::Point>> konturen_rs;
                std::vector<cv::Rect> boxen_rs;
                std::vector<cv::Rect> gefilterte_boxen_rs;
                cv::Mat frame_rs;
                for (size_t i = 0; i < datensatz.ergebnisse.size(); ++i) {
                    const auto& [konturen, boxen, gefilterte_boxen] = datensatz.ergebnisse[i];

                    if (!gefilterte_boxen.empty()) {
                        // Ergebnis sichern
                        konturen_rs = konturen;
                        boxen_rs = boxen;
                        gefilterte_boxen_rs = gefilterte_boxen;
                        frame_rs = datensatz.frames[i];
                        start_frame_index++;
                        break;
                    }
                }
                std::cout << "Datensatz erstellt bei Frame " << start_frame_index << endl;
                gen.debugDataset(konturen_rs,boxen_rs,gefilterte_boxen_rs,ref_frame_maskiert_gray,frame_rs, start_frame_index, referenz_frame_num, debug_path, config);
                std::string dateiname = std::format("frame_{:06}", start_frame_index);
                gen.speichereFuerYolo(frame_rs, gefilterte_boxen_rs, yolo_path, dateiname);
                // gen.speichereFuerCreateML(frame_rs, gefilterte_boxen_rs, ml_path, dateiname, CLASS_NAME);
                puffer_bewegung.clear();
                break;

            }
            genutzte_ergebnisse++;
        }
        if(neue_referenz_frame_num>referenz_frame_num) {
            referenz_frame_num=neue_referenz_frame_num;
            ref_frame_maskiert_gray = gen.ladeReferenzFrame(clip_path, referenz_frame_num, maske_bw,config);
        }
       
        // Neuen Frame-Puffer vorbereiten
        frame_puffer.clear();

        if (genutzte_ergebnisse < (int)frames.size()) {
            // hintere Frames behalten
            frame_puffer.insert(
                frame_puffer.end(),
                frames.begin() + genutzte_ergebnisse,
                frames.end()
            );
        }
        
        frame_counter+=genutzte_ergebnisse;
       
        if(frame_counter>end_frame_nummer) {
            break;
        }
    }

    cap.release();

    return 0;
}
