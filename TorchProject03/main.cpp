#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <sndfile.hh>

namespace fs = std::filesystem;

const int SAMPLE_RATE = 16000; // Hz
const int AUDIO_DURATION_SEC = 1;
const int INPUT_SIZE = SAMPLE_RATE * AUDIO_DURATION_SEC;


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


torch::Tensor load_audio(const std::string& path) {
    SndfileHandle file(path);
    if (file.channels() != 1 || file.samplerate() != SAMPLE_RATE) {
        throw std::runtime_error("Unsupported WAV format");
    }

    std::vector<float> samples(file.frames());
    file.readf(samples.data(), file.frames());

    if (samples.size() < INPUT_SIZE) {
        samples.resize(INPUT_SIZE, 0.0f); 
    }
    else if (samples.size() > INPUT_SIZE) {
        samples.resize(INPUT_SIZE);
    }

    return torch::from_blob(samples.data(), { 1, INPUT_SIZE }, torch::kFloat32).clone();
}

void load_dataset(const std::string& positive_dir, const std::string& negative_dir,
    std::vector<torch::Tensor>& data, std::vector<int64_t>& labels) {

    auto load_from_dir = [&](const std::string& dir_path, int64_t label) {
        for (const auto& entry : fs::recursive_directory_iterator(dir_path)) {
            if (!entry.is_regular_file()) continue;

            auto path = entry.path();
            if (path.extension() != ".wav") continue;

            try {
                data.push_back(load_audio(path.string()));
                labels.push_back(label);
            }
            catch (const std::exception& e) {
                std::cerr << "Ошибка при загрузке " << path << ": " << e.what() << "\n";
            }
        }
        };

    load_from_dir(positive_dir, 1); //pos
    load_from_dir(negative_dir, 0); //negg
}

int main() {
    torch::manual_seed(42);

    std::vector<torch::Tensor> data;
    std::vector<int64_t> labels;
    std::string path_yes = "D:\\Work\\media\\jarvis16000";
    std::string path_no = "D:\\Work\\media\\speech_commands_v0.02 (1).tar\\speech_commands_v0.02 (1)";

    load_dataset(path_yes, path_no, data, labels);

    auto x = torch::cat(data).to(torch::kFloat32);
    auto y = torch::tensor(labels, torch::kLong);

    Net model;
    torch::optim::Adam optimizer(model->parameters(), 0.001);

    const int epochs = 10;
    const int batch_size = 8;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "] - Started\n";
        model->train();
        float total_loss = 0.0;

        for (size_t i = 0; i < x.size(0); i += batch_size) {
            size_t end = std::min(int64_t(i + batch_size), x.size(0));
            auto batch_x = x.slice(0, i, end);
            auto batch_y = y.slice(0, i, end);

            optimizer.zero_grad();
            auto output = model->forward(batch_x);
            auto loss = torch::nll_loss(output, batch_y);
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
        }

        std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "] - Loss: " << total_loss << "\n";
    }

  
    torch::save(model, "model.pt");
    std::cout << "Model saved!\n";

    return 0;
}