#define _CRT_SECURE_NO_WARNINGS
#include "../wavreader.h" // Ваша библиотека для чтения WAV
#include <librosa/librosa.h> // Ваша C++ librosa

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <filesystem> // Для удобства работы с путями (опционально)

// --- Заголовки PyTorch ---
#include <torch/torch.h>
#include <torch/script.h> // Обязательно для torch::jit::load

using namespace std;


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


int main(int argc, char* argv[])
{
    // --- 1. Загрузка WAV и извлечение MFCC (Ваш существующий код) ---

    //const char* wav_path = "D:\\Work\\media\\jarvis16000\\0000_16k_mono.wav"; // Путь к аудио
    const char* wav_path = "D:\\Work\\media\\testfiles\\0002_16000mono.wav"; // Путь к аудио
    void* h_x = wav_read_open(wav_path);

    if (!h_x) {
        cerr << "Error opening wav file: " << wav_path << endl;
        return -1;
    }

    int format, channels, sr, bits_per_sample;
    unsigned int data_length;
    int res = wav_get_header(h_x, &format, &channels, &sr, &bits_per_sample, &data_length);
    if (!res)
    {
        cerr << "get ref header error: " << res << endl;
        wav_read_close(h_x);
        return -1;
    }

    int samples = data_length * 8 / bits_per_sample;
    std::vector<int16_t> tmp(samples);
    res = wav_read_data(h_x, reinterpret_cast<unsigned char*>(tmp.data()), data_length);
    if (res < 0)
    {
        cerr << "read wav file error: " << res << endl;
        wav_read_close(h_x);
        return -1;
    }
    wav_read_close(h_x); // Закрываем файл после чтения

    std::vector<float> x(samples);
    std::transform(tmp.begin(), tmp.end(), x.begin(),
        [](int16_t a) {
            return static_cast<float>(a) / 32767.f;
        });

    std::cout << "Audio loaded: " << wav_path << ", Sample rate: " << sr << "Hz" << std::endl;

    // Параметры MFCC (должны соответствовать тем, что использовались при обучении)
    int n_fft = 400;
    int n_hop = 160;
    int n_mel = 26; // Количество мел-фильтров
    int n_mfcc_out = 13; // Итоговое количество MFCC коэффициентов
    int fmin = 0;
    int fmax = 8000; // Или sr / 2, если не указано иное
    float power = 2.f;
    int dct_type = 2; // Обычно тип 2
    bool use_norm = true; // Использовать ли орто-нормализацию (соответствует norm='ortho')

    auto mfcc_start_time = std::chrono::system_clock::now();
    // Используем параметры, соответствующие Python librosa для лучшего совпадения
    // Обратите внимание на последние параметры: n_mfcc=13, dct_type=2, norm=true ('ortho')
    auto mfcc_vector = librosa::Feature::mfcc(x, sr, n_fft, n_hop, "hann", false, "reflect", power, n_mel, fmin, fmax, n_mfcc_out, use_norm, dct_type);
    auto mfcc_end_time = std::chrono::system_clock::now();
    auto mfcc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(mfcc_end_time - mfcc_start_time);

    if (mfcc_vector.empty() || mfcc_vector[0].empty()) {
        cerr << "Error: Failed to compute MFCC or MFCC vector is empty." << endl;
        return -1;
    }

    std::cout << "MFCC calculated. Shape (frames, coeffs): [" << mfcc_vector.size() << ", " << mfcc_vector[0].size() << "]" << std::endl;
    std::cout << "MFCC calculation time: " << mfcc_duration.count() << "ms" << std::endl;

    std::string model_path = "C:\\Users\\User\\source\\repos\\TorchProject03\\TorchProject03\\model_mfcc2_3classess_saved.pt";
    CnnNet model; // Создаем экземпляр вашей модели (нужно ее определение)

    try {
        // Загружаем сохраненное состояние (архитектуру и параметры) в созданный экземпляр
        torch::load(model, model_path);
        model->eval(); // Переводим модель в режим оценки
        std::cout << "Model loaded successfully using torch::load from: " << model_path << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model with torch::load:\n" << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) { // Добавим обработку std::exception на всякий случай
        std::cerr << "Error loading the model with torch::load:\n" << e.what() << std::endl;
        return -1;
    }

    // Определение устройства (CPU по умолчанию)
    torch::Device device(torch::kCPU);
    // if (torch::cuda::is_available()) { device = torch::kCUDA; } // Опционально GPU
    model->to(device); // Перемещаем модель на выбранное устройство




    // --- 3. Подготовка тензора MFCC для модели ---
    // Модель ожидает [Batch=1, Channels=13, SequenceLength=num_frames]
    // Наш mfcc_vector имеет форму [num_frames, n_mfcc=13]

        // нормализуем до T_FIXED по времени
    if (mfcc_vector.size() < T_FIXED) {
        // дополняем нулями
        mfcc_vector.resize(T_FIXED, std::vector<float>(n_mfcc_out, 0.0f));
    }
    else if (mfcc_vector.size() > T_FIXED) {
        // обрезаем
        mfcc_vector.resize(T_FIXED);
    }

    int num_frames = mfcc_vector.size();
    int n_mfcc = mfcc_vector[0].size();

    // 3.1. Сначала "выпрямим" вектор векторов в один плоский вектор
    std::vector<float> flat_mfcc;
    flat_mfcc.reserve(num_frames * n_mfcc);
    for (int i = 0; i < num_frames; ++i) {
        flat_mfcc.insert(flat_mfcc.end(), mfcc_vector[i].begin(), mfcc_vector[i].end());
    }

    // 3.2. Создаем тензор из плоского вектора данных (сначала форма [num_frames, n_mfcc])
    // Важно: from_blob НЕ владеет памятью. Убедитесь, что flat_mfcc существует, пока тензор используется,
    // или используйте .clone() для создания копии данных внутри тензора.
    torch::Tensor input_tensor = torch::from_blob(flat_mfcc.data(), { num_frames, n_mfcc }, torch::kFloat);

    // 3.3 Клонируем тензор, чтобы он владел своими данными
    input_tensor = input_tensor.clone();

    // 3.4. Транспонируем в [n_mfcc, num_frames]
    input_tensor = input_tensor.transpose(0, 1); // Теперь форма [13, num_frames]

    // 3.5. Добавляем измерение батча (batch dimension) в начало
    input_tensor = input_tensor.unsqueeze(0); // Теперь форма [1, 13, num_frames]

    // 3.6. Перемещаем тензор на то же устройство, что и модель
    input_tensor = input_tensor.to(device);

    std::cout << "Input tensor prepared for model. Shape: " << input_tensor.sizes() << std::endl;

    // +++ Используйте этот блок +++
    torch::Tensor output_tensor;
    try {
        torch::NoGradGuard no_grad; // Отключаем градиенты для инференса
        // Напрямую вызываем метод forward у нашего экземпляра модели
        output_tensor = model->forward(input_tensor); // Передаем тензор напрямую
    }
    catch (const c10::Error& e) {
        std::cerr << "Error during model inference:\n" << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) { // Добавим обработку std::exception
        std::cerr << "Error during model inference:\n" << e.what() << std::endl;
        return -1;
    }


    // --- 5. Обработка результата ---
    // output_tensor содержит логарифмы вероятностей (log_softmax) для каждого из 3 классов
    // Нам нужен индекс класса с максимальным значением

    // argmax вдоль оси классов (ось 1)
    torch::Tensor predicted_idx_tensor = torch::argmax(output_tensor, 1); // Результат будет тензором формы [1]
    int64_t predicted_idx = predicted_idx_tensor.item<int64_t>(); // Извлекаем значение индекса

    // Получаем вероятности (опционально, если нужны именно они)
    // torch::Tensor probabilities = torch::exp(output_tensor); // exp(log_softmax) -> softmax

    // Определяем имена классов (замените на ваши реальные имена)
    std::vector<std::string> class_names = { "noise", "jarvis", "turnon" };

    std::cout << "--- Inference Results ---" << std::endl;
    std::cout << "Raw model output (log probabilities): " << output_tensor << std::endl;
    // std::cout << "Probabilities: " << probabilities << std::endl; // Раскомментируйте, если нужны вероятности

    if (predicted_idx >= 0 && predicted_idx < class_names.size()) {
        std::cout << "Predicted class index: " << predicted_idx << std::endl;
        std::cout << "Predicted class name: " << class_names[predicted_idx] << std::endl;
    }
    else {
        std::cerr << "Error: Predicted index " << predicted_idx << " is out of bounds for class names." << std::endl;
        return -1;
    }

    return 0;
}