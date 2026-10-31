// Real-time sphere fiducial classification with the SVM trained on
// real_plus_synthetic.csv (train_export_model.py).
//
// Label convention of this model: 1 = Good, 0 = Bad.
// (The older sphere_model.onnx / spherefiducial_model.onnx use 0 = Good.)
//
// Per fiducial, every frame:
//   1. Depth check: PointZ outside 700-2800 mm -> Bad, model not used.
//   2. Scaler ONNX -> SVM ONNX: output_label (int64, 1 = Good) and
//      output_probability (float [N, 2], column 1 = P(Good)).
//   3. Smoothing: median P(Good) over the last WINDOW frames of that fiducial;
//      switch to Good at >= GOOD_ON, to Bad at <= BAD_ON, otherwise keep state.
//
// Modes:
//   ./sphere_realtime_inference --live [host] [port] [path]   poll the FusionTrack API
//   ./sphere_realtime_inference --manual                      type PointZ and Radius
//   ./sphere_realtime_inference --batch in.csv out.csv        CSV with PointZ/point_z and
//                                                             Radius/radius columns
// Model files are read from the working directory (run it from model/):
//   sphere_svm_scaler.onnx, sphere_svm_model.onnx
//
// Build (ONNX Runtime 1.20.1):
//   ORT=/path/to/onnxruntime-linux-x64-1.20.1
//   g++ -std=c++17 -O2 -I$ORT/include -L$ORT/lib -Wl,-rpath,$ORT/lib
//       -o sphere_realtime_inference sphere_realtime_inference.cpp -lonnxruntime

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <netdb.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

constexpr float Z_MIN = 700.0f;
constexpr float Z_MAX = 2800.0f;
constexpr size_t WINDOW = 5;
constexpr float GOOD_ON = 0.7f;
constexpr float BAD_ON = 0.3f;
constexpr long MAX_MISSING = 10;
constexpr int GOOD = 1;
constexpr int BAD = 0;

struct Fiducial {
    std::string key;
    float point_z;
    float radius;
};

struct Result {
    Fiducial fiducial;
    bool in_volume;
    int frame_label;  // model (or depth check) output for this frame
    float p_good;
    float smoothed_p_good;
    int state;        // smoothed decision
};

// ---------------------------------------------------------------- model
class SphereModel {
public:
    explicit SphereModel(Ort::Env& env)
        : scaler_(env, "sphere_svm_scaler.onnx", Ort::SessionOptions{}),
          model_(env, "sphere_svm_model.onnx", Ort::SessionOptions{}) {}

    // X: n rows of {PointZ, Radius}. Fills labels (1 = Good) and P(Good).
    void predict(const std::vector<float>& X, std::vector<int64_t>& labels,
                 std::vector<float>& p_good) {
        const int64_t n = static_cast<int64_t>(X.size() / 2);
        std::vector<int64_t> dims = {n, 2};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(X.data()), X.size(), dims.data(), dims.size());

        const char* scaler_in[] = {"input"};
        const char* scaler_out[] = {"variable"};
        auto scaled = scaler_.Run(Ort::RunOptions{nullptr}, scaler_in, &input, 1, scaler_out, 1);

        const char* model_in[] = {"input"};
        const char* model_out[] = {"output_label", "output_probability"};
        auto outputs = model_.Run(Ort::RunOptions{nullptr}, model_in, &scaled.front(), 1, model_out, 2);

        const int64_t* label_data = outputs[0].GetTensorData<int64_t>();  // int64, not float
        const float* prob_data = outputs[1].GetTensorData<float>();        // [n, 2]
        labels.assign(label_data, label_data + n);
        p_good.resize(n);
        for (int64_t i = 0; i < n; ++i) p_good[i] = prob_data[2 * i + 1];
    }

private:
    Ort::Session scaler_;
    Ort::Session model_;
};

// ---------------------------------------------------------------- classifier with smoothing
class SphereClassifier {
public:
    explicit SphereClassifier(SphereModel& model) : model_(model) {}

    std::vector<Result> classify_frame(const std::vector<Fiducial>& fiducials) {
        ++frame_;
        std::vector<Result> results;
        if (fiducials.empty()) return results;

        std::vector<float> X;
        std::vector<size_t> model_rows;
        results.resize(fiducials.size());
        for (size_t i = 0; i < fiducials.size(); ++i) {
            const Fiducial& f = fiducials[i];
            results[i] = {f, f.point_z >= Z_MIN && f.point_z <= Z_MAX, BAD, 0.0f, 0.0f, BAD};
            if (results[i].in_volume) {
                X.push_back(f.point_z);
                X.push_back(f.radius);
                model_rows.push_back(i);
            }
        }
        if (!model_rows.empty()) {
            std::vector<int64_t> labels;
            std::vector<float> p_good;
            model_.predict(X, labels, p_good);
            for (size_t j = 0; j < model_rows.size(); ++j) {
                results[model_rows[j]].frame_label = static_cast<int>(labels[j]);
                results[model_rows[j]].p_good = p_good[j];
            }
        }

        for (Result& r : results) {
            auto it = tracks_.find(r.fiducial.key);
            if (it == tracks_.end() || frame_ - it->second.last_frame > MAX_MISSING) {
                tracks_[r.fiducial.key] = Track{{}, r.frame_label, frame_};
                it = tracks_.find(r.fiducial.key);
            }
            Track& track = it->second;
            track.last_frame = frame_;
            track.probs.push_back(r.p_good);
            if (track.probs.size() > WINDOW) track.probs.pop_front();
            r.smoothed_p_good = median(track.probs);
            if (!r.in_volume) track.state = BAD;
            else if (r.smoothed_p_good >= GOOD_ON) track.state = GOOD;
            else if (r.smoothed_p_good <= BAD_ON) track.state = BAD;
            r.state = track.state;
        }
        return results;
    }

private:
    struct Track {
        std::deque<float> probs;
        int state;
        long last_frame;
    };

    static float median(const std::deque<float>& values) {
        std::vector<float> v(values.begin(), values.end());
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        return n % 2 ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
    }

    SphereModel& model_;
    std::map<std::string, Track> tracks_;
    long frame_ = 0;
};

// ---------------------------------------------------------------- HTTP GET (plain HTTP, no extra libraries)
bool http_get(const std::string& host, const std::string& port, const std::string& path,
              std::string& body) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) return false;
    int sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sock < 0 || connect(sock, res->ai_addr, res->ai_addrlen) != 0) {
        if (sock >= 0) close(sock);
        freeaddrinfo(res);
        return false;
    }
    freeaddrinfo(res);
    timeval timeout{5, 0};
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    // HTTP/1.0: the server closes the connection after the body, no chunking
    const std::string request = "GET " + path + " HTTP/1.0\r\nHost: " + host + "\r\n\r\n";
    send(sock, request.c_str(), request.size(), 0);
    std::string response;
    char buffer[8192];
    ssize_t n;
    while ((n = recv(sock, buffer, sizeof(buffer), 0)) > 0) response.append(buffer, n);
    close(sock);

    const size_t header_end = response.find("\r\n\r\n");
    if (header_end == std::string::npos || response.find(" 200") > response.find("\r\n")) return false;
    body = response.substr(header_end + 4);
    return true;
}

// ---------------------------------------------------------------- minimal JSON extraction
// Reads MarkerName, and point.z and Radius of each entry in every "FiducialList".
size_t matching_bracket(const std::string& s, size_t open) {
    const char open_c = s[open], close_c = open_c == '[' ? ']' : '}';
    int depth = 0;
    bool in_string = false;
    for (size_t i = open; i < s.size(); ++i) {
        if (s[i] == '"' && s[i - 1] != '\\') in_string = !in_string;
        if (in_string) continue;
        if (s[i] == open_c) ++depth;
        if (s[i] == close_c && --depth == 0) return i;
    }
    return std::string::npos;
}

bool number_after(const std::string& s, const std::string& key, size_t from, size_t to, float& value) {
    const size_t pos = s.find("\"" + key + "\"", from);
    if (pos == std::string::npos || pos >= to) return false;
    const size_t colon = s.find(':', pos);
    value = std::strtof(s.c_str() + colon + 1, nullptr);
    return true;
}

std::vector<Fiducial> parse_fiducials(const std::string& json) {
    std::vector<Fiducial> fiducials;
    size_t pos = 0;
    while ((pos = json.find("\"FiducialList\"", pos)) != std::string::npos) {
        std::string marker = "marker";
        const size_t name_pos = json.rfind("\"MarkerName\"", pos);
        if (name_pos != std::string::npos) {
            const size_t q1 = json.find('"', json.find(':', name_pos) + 1);
            marker = json.substr(q1 + 1, json.find('"', q1 + 1) - q1 - 1);
        }
        const size_t list_open = json.find('[', pos);
        const size_t list_close = matching_bracket(json, list_open);
        if (list_close == std::string::npos) break;
        size_t obj = list_open;
        int index = 0;
        while ((obj = json.find('{', obj + 1)) != std::string::npos && obj < list_close) {
            const size_t obj_close = matching_bracket(json, obj);
            const size_t point = json.find("\"point\"", obj);
            float z = 0.0f, radius = 0.0f;
            if (point < obj_close && number_after(json, "z", point, obj_close, z) &&
                number_after(json, "Radius", obj, obj_close, radius)) {
                fiducials.push_back({marker + ":" + std::to_string(index), z, radius});
            }
            ++index;
            obj = obj_close;
        }
        pos = list_close;
    }
    return fiducials;
}

// ---------------------------------------------------------------- modes
void print_result(const Result& r) {
    std::cout << "Fiducial " << r.fiducial.key << ": z=" << std::fixed << std::setprecision(1)
              << r.fiducial.point_z << " r=" << std::setprecision(3) << r.fiducial.radius
              << " P(Good)=" << std::setprecision(2) << r.smoothed_p_good << " -> "
              << (r.state == GOOD ? "Good (1)" : "Bad (0)")
              << (r.in_volume ? "" : " [outside 700-2800 mm]") << std::endl;
}

int run_live(SphereClassifier& classifier, const std::string& host, const std::string& port,
             const std::string& path) {
    std::cout << "Polling http://" << host << ":" << port << path << " (Ctrl+C to stop)" << std::endl;
    while (true) {
        std::string body;
        if (!http_get(host, port, path, body)) {
            std::cout << "Camera not responding..." << std::endl;
        } else {
            const auto fiducials = parse_fiducials(body);
            if (fiducials.empty()) std::cout << "No fiducials in frame" << std::endl;
            for (const Result& r : classifier.classify_frame(fiducials)) print_result(r);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int run_manual(SphereClassifier& classifier) {
    float point_z, radius;
    for (int entry = 1;; ++entry) {
        std::cout << "Enter PointZ and Radius (mm), or q to quit: ";
        if (!(std::cin >> point_z >> radius)) return 0;
        // Each manual reading is its own fiducial, so no smoothing across entries
        for (const Result& r : classifier.classify_frame({{"entry" + std::to_string(entry), point_z, radius}}))
            print_result(r);
    }
}

int run_batch(SphereClassifier& classifier, const std::string& in_path, const std::string& out_path) {
    std::ifstream in(in_path);
    std::ofstream out(out_path);
    if (!in || !out) {
        std::cerr << "Cannot open " << in_path << " or " << out_path << std::endl;
        return 1;
    }
    std::string line;
    // Accept Windows (\r\n) line endings
    auto read_line = [&in](std::string& l) {
        if (!std::getline(in, l)) return false;
        if (!l.empty() && l.back() == '\r') l.pop_back();
        return true;
    };
    read_line(line);
    std::vector<std::string> header;
    std::stringstream hs(line);
    for (std::string col; std::getline(hs, col, ',');) header.push_back(col);
    int z_col = -1, r_col = -1;
    for (int i = 0; i < static_cast<int>(header.size()); ++i) {
        if (header[i] == "PointZ" || header[i] == "point_z") z_col = i;
        if (header[i] == "Radius" || header[i] == "radius") r_col = i;
    }
    if (z_col < 0 || r_col < 0) {
        std::cerr << "Need PointZ/point_z and Radius/radius columns" << std::endl;
        return 1;
    }
    // Each row is classified on its own (no smoothing), to compare with Python
    out << "point_z,radius,in_volume,label,p_good\n" << std::setprecision(9);
    int rows = 0;
    while (read_line(line)) {
        std::vector<std::string> cells;
        std::stringstream ls(line);
        for (std::string cell; std::getline(ls, cell, ',');) cells.push_back(cell);
        const float z = std::stof(cells[z_col]), radius = std::stof(cells[r_col]);
        const Result r = classifier.classify_frame({{"row" + std::to_string(rows), z, radius}})[0];
        out << z << "," << radius << "," << r.in_volume << "," << r.frame_label << "," << r.p_good << "\n";
        ++rows;
    }
    std::cout << "Classified " << rows << " rows -> " << out_path << std::endl;
    return 0;
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "--manual";
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SphereFiducial");
    SphereModel model(env);
    SphereClassifier classifier(model);
    std::cout << "Sphere fiducial SVM loaded. Labels: 1 = Good, 0 = Bad. Volume 700-2800 mm." << std::endl;

    if (mode == "--live") {
        return run_live(classifier, argc > 2 ? argv[2] : "127.0.0.1", argc > 3 ? argv[3] : "8081",
                        argc > 4 ? argv[4] : "/GetCameraData");
    }
    if (mode == "--batch" && argc > 3) return run_batch(classifier, argv[2], argv[3]);
    if (mode == "--manual") return run_manual(classifier);
    std::cerr << "Usage: " << argv[0] << " --live [host] [port] [path] | --manual | --batch in.csv out.csv"
              << std::endl;
    return 1;
}
