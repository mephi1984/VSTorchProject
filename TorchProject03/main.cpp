#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <random>
#include <sndfile.hh>

namespace fs = std::filesystem;

const int SAMPLE_RATE = 16000; // Hz
const int AUDIO_DURATION_SEC = 1;
const int INPUT_SIZE = SAMPLE_RATE * AUDIO_DURATION_SEC;
const int64_t MAX_NEGATIVES = 500; // ограничим число негативных примеров

struct NetImpl : torch::nn::Module {
    torch::nn::Linear fc1, fc2, fc3;

    NetImpl() :
        fc1(INPUT_SIZE, 512),
        fc2(512, 128),
        fc3(128, 2) {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return torch::log_softmax(fc3->forward(x), /*dim=*/1);
    }
};
TORCH_MODULE(Net);

// Загрузка и нормализация аудио
torch::Tensor load_audio(const std::string& path) {
    SndfileHandle file(path);
    if (file.channels() != 1 || file.samplerate() != SAMPLE_RATE) {
        throw std::runtime_error("Unsupported WAV format");
    }

    std::vector<float> samples(file.frames());
    file.readf(samples.data(), file.frames());

    if (samples.size() < INPUT_SIZE)
        samples.resize(INPUT_SIZE, 0.0f);
    else if (samples.size() > INPUT_SIZE)
        samples.resize(INPUT_SIZE);

    // 🔽 Нормализация
    for (auto& sample : samples) {
        sample = std::clamp(sample / 32768.0f, -1.0f, 1.0f);
    }

    return torch::from_blob(samples.data(), { 1, INPUT_SIZE }, torch::kFloat32).clone();
}

// Загрузка данных из директорий
void load_dataset(const std::string& positive_dir, const std::string& negative_dir,
    std::vector<torch::Tensor>& data, std::vector<int64_t>& labels) {
    int64_t neg_count = 0;
    auto load_from_dir = [&](const std::string& dir_path, int64_t label) {
        for (const auto& entry : fs::recursive_directory_iterator(dir_path)) {
            if (!entry.is_regular_file()) continue;
            auto path = entry.path();
            if (path.extension() != ".wav") continue;

            if (label == 0 && neg_count >= MAX_NEGATIVES) break;

            try {
                data.push_back(load_audio(path.string()));
                labels.push_back(label);
                if (label == 0) ++neg_count;
            }
            catch (const std::exception& e) {
                std::cerr << "Ошибка при загрузке " << path << ": " << e.what() << "\n";
            }
        }
        };

    load_from_dir(positive_dir, 1);
    load_from_dir(negative_dir, 0);

    std::cout << "Данные загружены: " << data.size() << " примеров ("
        << std::count(labels.begin(), labels.end(), 1) << " позитивных, "
        << std::count(labels.begin(), labels.end(), 0) << " негативных)\n";
}

int main() {
    torch::manual_seed(777);

    std::vector<torch::Tensor> data;
    std::vector<int64_t> labels;
    std::string path_yes = "D:\\Work\\media\\jarvis16000";
    std::string path_no = "D:\\Work\\media\\speech_commands_v0.02 (1).tar\\speech_commands_v0.02 (1)";

    load_dataset(path_yes, path_no, data, labels);

    // 🔀 Перемешивание
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{ std::random_device{}() });

    std::vector<torch::Tensor> x_data;
    std::vector<int64_t> y_data;
    for (size_t i : indices) {
        x_data.push_back(data[i]);
        y_data.push_back(labels[i]);
    }



    // 🧪 Train/Validation Split
    size_t train_size = static_cast<size_t>(x_data.size() * 0.8);
    auto x_train = torch::cat(std::vector<torch::Tensor>(x_data.begin(), x_data.begin() + train_size)).to(torch::kFloat32);
    auto y_train = torch::tensor(std::vector<int64_t>(y_data.begin(), y_data.begin() + train_size), torch::kLong);
    auto x_val = torch::cat(std::vector<torch::Tensor>(x_data.begin() + train_size, x_data.end())).to(torch::kFloat32);
    auto y_val = torch::tensor(std::vector<int64_t>(y_data.begin() + train_size, y_data.end()), torch::kLong);

    Net model;
    torch::optim::Adam optimizer(model->parameters(), 0.0001); // 🔽 Learning Rate

    const int epochs = 10;
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
            auto loss = torch::nll_loss(output, batch_y);
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
        }

        // 📊 Валидация
        model->eval();
        int correct = 0;
        for (int i = 0; i < x_val.size(0); ++i) {
            auto input = x_val[i];
            auto output = model->forward(input.unsqueeze(0));
            auto pred = output.argmax(1).item<int>();
            if (pred == y_val[i].item<int>()) correct++;
        }
        float acc = static_cast<float>(correct) / x_val.size(0);

        std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "] "
            << "- Train Loss: " << total_loss
            << ", Val Accuracy: " << acc * 100 << "%\n";
    }

    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (int i = 0; i < x_val.size(0); ++i) {
        auto output = model->forward(x_val[i].unsqueeze(0));
        int pred = output.argmax(1).item<int>();
        int label = y_val[i].item<int>();
        if (label == 1 && pred == 1) tp++;
        else if (label == 0 && pred == 0) tn++;
        else if (label == 0 && pred == 1) fp++;
        else if (label == 1 && pred == 0) fn++;
    }
    std::cout << "TP=" << tp << " FP=" << fp << " FN=" << fn << " TN=" << tn << "\n";

    torch::save(model, "model.pt");
    std::cout << "Model saved!\n";

    return 0;
}