#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>

namespace fs = std::filesystem;

const int MFCC_DIM = 13; // 13 коэффициентов MFCC де-дельта-дельта
const int64_t MAX_NEGATIVES = 500;

struct NetImpl : torch::nn::Module {
    torch::nn::Linear fc1, fc2, fc3, fc4;
    torch::nn::Dropout dropout1, dropout2;

    NetImpl()
        : fc1(MFCC_DIM, 128),
        fc2(128, 64),
        fc3(64, 32),
        fc4(32, 2),
        dropout1(0.3),
        dropout2(0.3) {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        register_module("fc4", fc4);
        register_module("dropout1", dropout1);
        register_module("dropout2", dropout2);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = dropout1->forward(x);
        x = torch::relu(fc2->forward(x));
        x = dropout2->forward(x);
        x = torch::relu(fc3->forward(x));
        return torch::log_softmax(fc4->forward(x), /*dim=*/1);
    }
};
TORCH_MODULE(Net);

// Загрузка и усреднение MFCC из .csv
torch::Tensor load_mfcc_csv(const std::string& path) {
    std::ifstream file(path);
    std::string line;

    // пропустить заголовок
    std::getline(file, line);

    std::vector<std::vector<float>> frames;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::getline(ss, token, ';'); // frameIndex
        std::getline(ss, token, ';'); // frameTime

        std::vector<float> coeffs;
        while (std::getline(ss, token, ';')) {
            coeffs.push_back(std::stof(token));
        }

        if (coeffs.size() == MFCC_DIM)
            frames.push_back(coeffs);
    }

    if (frames.empty()) {
        throw std::runtime_error("Empty or invalid CSV: " + path);
    }

    torch::Tensor avg = torch::zeros({ MFCC_DIM });
    for (const auto& frame : frames) {
        for (int i = 0; i < MFCC_DIM; ++i) {
            avg[i] += frame[i];
        }
    }
    avg /= frames.size() + 0.0;
    return avg.unsqueeze(0);  // [1, MFCC_DIM]
}

// Загрузка CSV-датасета
void load_dataset(const std::string& positive_dir, const std::string& negative_dir,
    std::vector<torch::Tensor>& data, std::vector<int64_t>& labels) {

    int64_t neg_count = 0;

    auto load_from_dir = [&](const std::string& dir_path, int64_t label) {
        for (const auto& entry : fs::recursive_directory_iterator(dir_path)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".csv") continue;

            if (label == 0 && neg_count >= MAX_NEGATIVES) break;

            try {
                auto tensor = load_mfcc_csv(entry.path().string());
                data.push_back(tensor);
                labels.push_back(label);
                if (label == 0) ++neg_count;
            }
            catch (const std::exception& e) {
                std::cerr << "Ошибка при загрузке " << entry.path() << ": " << e.what() << "\n";
            }
        }
        };

    load_from_dir(positive_dir, 1);
    load_from_dir(negative_dir, 0);

    std::cout << "Загружено: " << data.size()
        << " примеров (" << std::count(labels.begin(), labels.end(), 1)
        << " позитивных, " << std::count(labels.begin(), labels.end(), 0) << " негативных)\n";
}

// gamma = 2.0, alpha = class weight tensor
torch::Tensor focal_loss(const torch::Tensor& input, const torch::Tensor& target, const torch::Tensor& alpha, float gamma = 2.0f) {
    auto logpt = torch::nll_loss(input, target, alpha, torch::Reduction::None);
    auto pt = torch::exp(-logpt);
    auto loss = (1 - pt).pow(gamma) * logpt;
    return loss.mean();
}

int main() {
    torch::manual_seed(777);

    std::vector<torch::Tensor> data;
    std::vector<int64_t> labels;

    //std::string path_yes = "D:\\Work\\mfcc\\yes"; // директория с позитивными CSV
    //std::string path_no = "D:\\Work\\mfcc\\no";  // директория с негативными CSV

    std::string path_yes = "D:\\Work\\media\\jarvis16000_mfcc";
    std::string path_no = "D:\\Work\\media\\speech_commands_v0.02_mfcc";


    load_dataset(path_yes, path_no, data, labels);

    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{ std::random_device{}() });

    std::vector<torch::Tensor> x_data;
    std::vector<int64_t> y_data;
    for (size_t i : indices) {
        x_data.push_back(data[i]);
        y_data.push_back(labels[i]);
    }



    size_t train_size = static_cast<size_t>(x_data.size() * 0.8);

    std::vector<torch::Tensor> x_train_vec(x_data.begin(), x_data.begin() + train_size);
    auto x_train = torch::cat(x_train_vec).to(torch::kFloat32);

    std::vector<torch::Tensor> x_val_vec(x_data.begin() + train_size, x_data.end());
    auto x_val = torch::cat(x_val_vec).to(torch::kFloat32);

    std::vector<int64_t> y_train_vec(y_data.begin(), y_data.begin() + train_size);
    auto y_train = torch::tensor(y_train_vec, torch::kLong);

    std::vector<int64_t> y_val_vec(y_data.begin() + train_size, y_data.end());
    auto y_val = torch::tensor(y_val_vec, torch::kLong);
    
    Net model;
    torch::optim::Adam optimizer(model->parameters(), 0.0001);


    // Пример: "Привет" — вес 5.0, "Не Привет" — вес 1.0
    torch::Tensor class_weights = torch::tensor({ 1.0, 5.0 }, torch::kFloat32);

    const int epochs = 100;
    const int batch_size = 8;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        model->train();
        float total_loss = 0.0;

        for (size_t i = 0; i < x_train.size(0); i += batch_size) {
            size_t end = std::min(int64_t(i + batch_size), x_train.size(0));
            auto batch_x = x_train.slice(0, i, end);
            auto batch_y = y_train.slice(0, i, end);

            optimizer.zero_grad();
            auto output = model->forward(batch_x);
            auto loss = focal_loss(output, batch_y, class_weights);
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
        }

        // Валидация
        model->eval();
        int correct = 0;
        for (int i = 0; i < x_val.size(0); ++i) {
            auto pred = model->forward(x_val[i].unsqueeze(0)).argmax(1).item<int>();
            if (pred == y_val[i].item<int>()) correct++;
        }

        float acc = static_cast<float>(correct) / x_val.size(0);
        std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "] "
            << "- Train Loss: " << total_loss
            << ", Val Accuracy: " << acc * 100 << "%\n";
    }

    torch::save(model, "model_mfcc.pt");
    std::cout << "Model saved!\n";

    return 0;
}