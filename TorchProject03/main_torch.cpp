#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>

namespace fs = std::filesystem;

const int MFCC_DIM = 13;
const int64_t MAX_NEGATIVES_POSITIVES = 1300;

struct CnnNetImpl : torch::nn::Module {
    torch::nn::Sequential conv_layers;
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };

    CnnNetImpl() {
        conv_layers = torch::nn::Sequential(
            torch::nn::Conv1d(torch::nn::Conv1dOptions(13, 64, 5).stride(1).padding(2)),
            torch::nn::BatchNorm1d(64),
            torch::nn::ReLU(),
            torch::nn::Dropout(0.2),

            torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm1d(128),
            torch::nn::ReLU(),
            torch::nn::Dropout(0.2),

            torch::nn::Conv1d(torch::nn::Conv1dOptions(128, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm1d(256),
            torch::nn::ReLU(),
            torch::nn::Dropout(0.3)
        );

        fc1 = torch::nn::Linear(256, 64);
        fc2 = torch::nn::Linear(64, 5);  

        register_module("conv_layers", conv_layers);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv_layers->forward(x);                 
        x = torch::adaptive_avg_pool1d(x, 1).squeeze(2); 
        x = torch::relu(fc1->forward(x));            
        x = torch::log_softmax(fc2->forward(x), 1); 
        return x;
    }
};
TORCH_MODULE(CnnNet);

constexpr int T_FIXED = 140; 

torch::Tensor load_mfcc_csv(const std::string& path) {
    std::ifstream file(path);
    std::string line;

    std::vector<std::vector<float>> frames;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<float> coeffs;
        while (std::getline(ss, token, ';')) {
            coeffs.push_back(std::stof(token));
        }

        if (coeffs.size() == MFCC_DIM)
            frames.push_back(coeffs);
    }

    if (frames.size() < T_FIXED) {
        frames.resize(T_FIXED, std::vector<float>(MFCC_DIM, 0.0f));
    }
    else if (frames.size() > T_FIXED) {
        frames.resize(T_FIXED);
    }

    torch::Tensor t = torch::zeros({ T_FIXED, MFCC_DIM });
    for (size_t i = 0; i < T_FIXED; ++i)
        for (int j = 0; j < MFCC_DIM; ++j)
            t[i][j] = frames[i][j];

    return t.transpose(0, 1).unsqueeze(0).clone();
}

void load_dataset(const std::string& path_privet, const std::string& path_vklyuchay, const std::string& path_codered, const std::string& path_wakeup, const std::string& path_noise,
    std::vector<torch::Tensor>& data, std::vector<int64_t>& labels) {

    auto load_from_dir = [&](const std::string& dir_path, int64_t label, int64_t max_count = -1) {
        int64_t neg_count = 0;
        for (const auto& entry : fs::recursive_directory_iterator(dir_path)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".csv") continue;
            if (max_count > 0 && neg_count >= max_count) break;

            try {
                auto tensor = load_mfcc_csv(entry.path().string());
                data.push_back(tensor);
                labels.push_back(label);
                ++neg_count;
            }
            catch (const std::exception& e) {
                std::cerr << "Ошибка при загрузке " << entry.path() << ": " << e.what() << "\n";
            }
        }
        };

    load_from_dir(path_privet, 1, MAX_NEGATIVES_POSITIVES);
    load_from_dir(path_vklyuchay, 2, MAX_NEGATIVES_POSITIVES);
    load_from_dir(path_codered, 3, MAX_NEGATIVES_POSITIVES);
    load_from_dir(path_wakeup, 4, MAX_NEGATIVES_POSITIVES);
    load_from_dir(path_noise, 0, MAX_NEGATIVES_POSITIVES);

    
    std::cout << "Загружено: " << data.size()
        << " (Привет=" << std::count(labels.begin(), labels.end(), 1)
        << ", Включай=" << std::count(labels.begin(), labels.end(), 2)
        << ", codered=" << std::count(labels.begin(), labels.end(), 3)
        << ", wakeup=" << std::count(labels.begin(), labels.end(), 4)
        << ", Остальное=" << std::count(labels.begin(), labels.end(), 0) << ")\n";
}

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

        std::string path_jarvis = "D:\\Work\\media\\2025-04-07try2\\jarvis_all_aug_mfcc";
        std::string path_turn = "D:\\Work\\media\\2025-04-07try2\\turnon_all_aug_mfcc";
        std::string path_codered = "D:\\Work\\media\\2025-04-07try2\\protocol_all_aug_mfcc";
        std::string path_wakeup = "D:\\Work\\media\\2025-04-07try2\\wakeup_all_aug_mfcc";

        std::string path_no = "D:\\Work\\media\\2025-04-07try2\\noise_wav_aug_mfcc";
        

        load_dataset(path_jarvis, path_turn, path_codered, path_wakeup, path_no, data, labels);

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

        torch::Tensor class_weights = torch::tensor({ 1.0, 1.0, 1.0, 1.0, 1.0 }, torch::kFloat32); 

        const int epochs = 30;
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

            model->eval();
            int correct = 0;
            std::vector<std::vector<int>> confusion(5, std::vector<int>(5, 0));  // [real][pred]

            for (int i = 0; i < x_val.size(0); ++i) {
                auto output = model->forward(x_val[i].unsqueeze(0));
                int pred = output.argmax(1).item<int>();
                int label = y_val[i].item<int>();

                if (pred == label) correct++;
                confusion[label][pred]++;
            }

            float acc = static_cast<float>(correct) / x_val.size(0);

            std::cout << "Confusion Matrix:\n";
            std::cout << "   P0  P1  P2  P3  P4\n";
            for (int real = 0; real < 5; ++real) {
                std::cout << "R" << real << " ";
                for (int pred = 0; pred < 5; ++pred) {
                    std::cout << std::setw(4) << confusion[real][pred];
                }
                std::cout << "\n";
            }

            std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "] "
                << "- Train Loss: " << total_loss
                << ", Val Accuracy: " << acc * 100 << "%\n";
        }

        torch::save(model, "model_mfcc2_5classes_tf_mega_aug200.pt");
        std::cout << "Model saved!\n";
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
    return 0;
}