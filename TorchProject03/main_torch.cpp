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

struct CnnNetImpl : torch::nn::Module {
    torch::nn::Conv1d conv1{ nullptr }, conv2{ nullptr };
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };

    CnnNetImpl()
        : conv1(torch::nn::Conv1dOptions(13, 32, 5).stride(1).padding(2)),  // [B, 13, T] → [B, 32, T]
        conv2(torch::nn::Conv1dOptions(32, 64, 3).stride(1).padding(1)), // [B, 64, T]
        fc1(64, 32),
        fc2(32, 3)
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = torch::adaptive_avg_pool1d(x, 1).squeeze(2); // [B, 64, T] → [B, 64]
        x = torch::relu(fc1->forward(x));
        return torch::log_softmax(fc2->forward(x), 1);
    }
};
TORCH_MODULE(CnnNet);

constexpr int T_FIXED = 100; // число временных кадров

torch::Tensor load_mfcc_csv(const std::string& path) {
    std::ifstream file(path);
    std::string line;

    //std::getline(file, line); // skip header
    std::vector<std::vector<float>> frames;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        //std::getline(ss, token, ';'); // frameIndex
        //std::getline(ss, token, ';'); // frameTime

        std::vector<float> coeffs;
        while (std::getline(ss, token, ';')) {
            coeffs.push_back(std::stof(token));
        }

        if (coeffs.size() == MFCC_DIM)
            frames.push_back(coeffs);
    }

    // нормализуем до T_FIXED по времени
    if (frames.size() < T_FIXED) {
        // дополняем нулями
        frames.resize(T_FIXED, std::vector<float>(MFCC_DIM, 0.0f));
    }
    else if (frames.size() > T_FIXED) {
        // обрезаем
        frames.resize(T_FIXED);
    }

    // [T, 13] → [1, 13, T]
    torch::Tensor t = torch::zeros({ T_FIXED, MFCC_DIM });
    for (size_t i = 0; i < T_FIXED; ++i)
        for (int j = 0; j < MFCC_DIM; ++j)
            t[i][j] = frames[i][j];

    return t.transpose(0, 1).unsqueeze(0).clone(); // [1, 13, T]
}

// Загрузка CSV-датасета
void load_dataset(const std::string& path_privet, const std::string& path_vklyuchay, const std::string& path_noise,
    std::vector<torch::Tensor>& data, std::vector<int64_t>& labels) {

    int64_t neg_count = 0;

    auto load_from_dir = [&](const std::string& dir_path, int64_t label, int64_t max_count = -1) {
        for (const auto& entry : fs::recursive_directory_iterator(dir_path)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".csv") continue;
            if (max_count > 0 && label == 0 && neg_count >= max_count) break;

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

    load_from_dir(path_privet, 1);
    load_from_dir(path_vklyuchay, 2);
    load_from_dir(path_noise, 0, MAX_NEGATIVES);

    std::cout << "Загружено: " << data.size()
        << " (Привет=" << std::count(labels.begin(), labels.end(), 1)
        << ", Включай=" << std::count(labels.begin(), labels.end(), 2)
        << ", Остальное=" << std::count(labels.begin(), labels.end(), 0) << ")\n";
}

// gamma = 2.0, alpha = class weight tensor
torch::Tensor focal_loss(const torch::Tensor& input, const torch::Tensor& target, const torch::Tensor& alpha, float gamma = 2.0f) {
    auto logpt = torch::nll_loss(input, target, alpha, torch::Reduction::None);
    auto pt = torch::exp(-logpt);
    auto loss = (1 - pt).pow(gamma) * logpt;
    return loss.mean();
}

int main() {
    try
    {
        torch::manual_seed(777);

        std::vector<torch::Tensor> data;
        std::vector<int64_t> labels;

        //std::string path_yes = "D:\\Work\\mfcc\\yes"; // директория с позитивными CSV
        //std::string path_no = "D:\\Work\\mfcc\\no";  // директория с негативными CSV

        std::string path_jarvis = "D:\\Work\\media\\jarvis16000_mfcc2";
        std::string path_turn = "D:\\Work\\media\\ВключиРок16000_mfcc2";

        std::string path_no = "D:\\Work\\media\\speech_commands_v0.02_mfcc2";
        

        load_dataset(path_jarvis, path_turn, path_no, data, labels);

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

        CnnNet model;

        torch::optim::Adam optimizer(model->parameters(), 0.0001);

        torch::Tensor class_weights = torch::tensor({ 1.0, 5.0, 5.0 }, torch::kFloat32);  // [шум, привет, включай]

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
            std::vector<std::vector<int>> confusion(3, std::vector<int>(3, 0));  // [real][pred]

            for (int i = 0; i < x_val.size(0); ++i) {
                auto output = model->forward(x_val[i].unsqueeze(0));
                int pred = output.argmax(1).item<int>();
                int label = y_val[i].item<int>();

                if (pred == label) correct++;
                confusion[label][pred]++;
            }

            float acc = static_cast<float>(correct) / x_val.size(0);

            // Печать confusion matrix
            std::cout << "Confusion Matrix:\n";
            std::cout << "   P0  P1  P2\n";
            for (int real = 0; real < 3; ++real) {
                std::cout << "R" << real << " ";
                for (int pred = 0; pred < 3; ++pred) {
                    std::cout << std::setw(4) << confusion[real][pred];
                }
                std::cout << "\n";
            }

            std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "] "
                << "- Train Loss: " << total_loss
                << ", Val Accuracy: " << acc * 100 << "%\n";
        }

        torch::save(model, "model_mfcc2_3classes.pt");
        std::cout << "Model saved!\n";
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
    return 0;
}