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

#include <portaudio.h>
#include <list>

// Функция для обработки данных (будет определена позже)
void processData(const std::vector<float>& data);

// Структура для хранения аудио данных
struct AudioData {
    std::vector<float> buffer;
    int samplesNeeded = 160;
};

using namespace std;


const int MFCC_DIM = 13; // 13 коэффициентов MFCC де-дельта-дельта
const int64_t MAX_NEGATIVES = 500;


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
        fc2 = torch::nn::Linear(64, 5);  // 5 классов

        register_module("conv_layers", conv_layers);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv_layers->forward(x);                      // [B, 13, T] → [B, 256, T]
        x = torch::adaptive_avg_pool1d(x, 1).squeeze(2);  // [B, 256]
        x = torch::relu(fc1->forward(x));                 // [B, 64]
        x = torch::log_softmax(fc2->forward(x), 1);       // [B, 5]
        return x;
    }
};
TORCH_MODULE(CnnNet);

constexpr int T_FIXED = 140; // число временных кадров

CnnNet model;
torch::Device device(torch::kCPU);



// Callback-функция PortAudio
static int audioCallback(const void* inputBuffer, void* outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags statusFlags,
    void* userData) {
    AudioData* data = static_cast<AudioData*>(userData);
    const int16_t* samples = static_cast<const int16_t*>(inputBuffer);

    (void)outputBuffer; // Предотвращаем предупреждение о неиспользуемой переменной

    for (unsigned long i = 0; i < framesPerBuffer; ++i) {
        // Конвертируем 16-bit в float (-1.0 до 1.0)
        float sample = static_cast<float>(samples[i]) / 32768.0f;
        data->buffer.push_back(sample);

        // Когда накопилось достаточно сэмплов, обрабатываем их
        if (data->buffer.size() >= data->samplesNeeded) {
            processData(data->buffer);
            data->buffer.clear();
        }
    }

    return paContinue;
}


std::mutex mfcc_buffer_mutex;
std::list<std::vector<float>> mfcc_buffer;
volatile bool shouldExit = false;

void threadFunction();

// Функция для вывода списка устройств и выбора микрофона
PaDeviceIndex selectInputDevice() {
    int numDevices = Pa_GetDeviceCount();
    if (numDevices < 1) {
        std::cerr << "No audio devices found!" << std::endl;
        return paNoDevice;
    }

    std::cout << "Available input devices:" << std::endl;

    const PaDeviceInfo* deviceInfo;
    PaDeviceIndex defaultInput = Pa_GetDefaultInputDevice();

    for (int i = 0; i < numDevices; i++) {
        deviceInfo = Pa_GetDeviceInfo(i);
        if (deviceInfo->maxInputChannels > 0) {
            std::cout << i << ". " << deviceInfo->name;
            if (i == defaultInput) std::cout << " (default)";
            std::cout << " [in:" << deviceInfo->maxInputChannels
                << " out:" << deviceInfo->maxOutputChannels << "]" << std::endl;
        }
    }

    PaDeviceIndex selectedDevice;
    while (true) {
        std::cout << "Select input device (0-" << numDevices - 1 << "): ";
        std::cin >> selectedDevice;

        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a number." << std::endl;
            continue;
        }

        if (selectedDevice < 0 || selectedDevice >= numDevices) {
            std::cout << "Device index out of range. Try again." << std::endl;
            continue;
        }

        deviceInfo = Pa_GetDeviceInfo(selectedDevice);
        if (deviceInfo->maxInputChannels < 1) {
            std::cout << "Selected device has no input channels. Try again." << std::endl;
            continue;
        }

        break;
    }

    return selectedDevice;
}

int main() {
    int n_mfcc_out = 13;


    //mfcc_buffer.resize(T_FIXED, std::vector<float>(n_mfcc_out, 0.0f));
    

    std::string model_path = "C:\\Users\\User\\source\\repos\\TorchProject03\\TorchProject03\\model_mfcc2_5classes_tf_mega_aug200.pt";

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

    model->to(device);

    PaError err;
    AudioData audioData;

    // Инициализация PortAudio
    err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return 1;
    }


    // Настройка параметров потока
    //PaStreamParameters inputParameters;
    /*
    inputParameters.device = Pa_GetDefaultInputDevice();
    if (inputParameters.device == paNoDevice) {
        std::cerr << "No default input device" << std::endl;
        Pa_Terminate();
        return 1;
    }*/

    PaDeviceIndex inputDevice = selectInputDevice();
    if (inputDevice == paNoDevice) {
        Pa_Terminate();
        return 1;
    }

    // Настройка параметров потока
    PaStreamParameters inputParameters;
    inputParameters.device = inputDevice;
    inputParameters.channelCount = 1;       // Моно
    inputParameters.sampleFormat = paInt16; // 16-bit целые числа
    inputParameters.suggestedLatency =
        Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;

    // Открываем поток
    PaStream* stream;
    err = Pa_OpenStream(&stream,
        &inputParameters,
        nullptr,    // Нет вывода
        16000,      // Частота дискретизации 16 кГц
        160,        // Кадры на буфер
        paClipOff,  // Мы не будем выводить звук
        audioCallback,
        &audioData);

    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        Pa_Terminate();
        return 1;
    }

    // Запускаем поток
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }


    //std::thread t(&threadFunction);

    std::cout << "Recording from "
        << Pa_GetDeviceInfo(inputParameters.device)->name
        << "\nPress Enter to stop..." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get(); // Ждем нажатия Enter

    shouldExit = true;

    // Останавливаем и закрываем поток
    err = Pa_StopStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
    }

    Pa_CloseStream(stream);
    Pa_Terminate();

    ///t.join();

    return 0;
}


void processData(const std::vector<float>& data) {

    int sr = 16000;
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

    //std::cout << "processData" << std::endl;


    static std::vector<float> accumulatedData;

    accumulatedData.insert(accumulatedData.end(), data.begin(), data.end());

    if (accumulatedData.size() < n_fft)
    {
        return;
    }

    static std::vector<float> portionData;

    portionData.resize(n_fft);

    std::copy(accumulatedData.begin(), accumulatedData.begin() + n_fft, portionData.begin());

    accumulatedData.erase(accumulatedData.begin(), accumulatedData.begin() + n_fft);

    auto part_mfcc_vector = librosa::Feature::mfcc(portionData, sr, n_fft, n_hop, "hann", false, "reflect", power, n_mel, fmin, fmax, n_mfcc_out, use_norm, dct_type);

    if (part_mfcc_vector.size() != 1)
    {
        throw std::runtime_error("mfcc data is not added");
    }

    //mfcc_buffer_mutex.lock();

    


    mfcc_buffer.insert(mfcc_buffer.end(), part_mfcc_vector[0]);

    if (mfcc_buffer.size() < T_FIXED)
    {
        return;
    }


    while (mfcc_buffer.size() > T_FIXED)
    {
        mfcc_buffer.erase(mfcc_buffer.begin());
    }

    //mfcc_buffer_mutex.unlock();

    int n_mfcc = 13;
    int num_frames = T_FIXED;



    std::array<float, 13 * T_FIXED> flat_mfcc;

    auto itr = mfcc_buffer.begin();
    for (size_t i = 0; i < T_FIXED; i++)
    {
        std::copy(itr->begin(), itr->end(), flat_mfcc.begin() + (i * 13));
        itr++;
    }

    torch::Tensor input_tensor = torch::from_blob(flat_mfcc.data(), { num_frames, n_mfcc }, torch::kFloat);

    // 3.3 Клонируем тензор, чтобы он владел своими данными
    input_tensor = input_tensor.clone();

    // 3.4. Транспонируем в [n_mfcc, num_frames]
    input_tensor = input_tensor.transpose(0, 1); // Теперь форма [13, num_frames]

    // 3.5. Добавляем измерение батча (batch dimension) в начало
    input_tensor = input_tensor.unsqueeze(0); // Теперь форма [1, 13, num_frames]

    // 3.6. Перемещаем тензор на то же устройство, что и модель
    input_tensor = input_tensor.to(device);

    //std::cout << "Input tensor prepared for model. Shape: " << input_tensor.sizes() << std::endl;

    // +++ Используйте этот блок +++
    torch::Tensor output_tensor;
    try {
        torch::NoGradGuard no_grad; // Отключаем градиенты для инференса
        // Напрямую вызываем метод forward у нашего экземпляра модели
        output_tensor = model->forward(input_tensor); // Передаем тензор напрямую
    }
    catch (const c10::Error& e) {
        std::cerr << "Error during model inference:\n" << e.what() << std::endl;
        return;
    }
    catch (const std::exception& e) { // Добавим обработку std::exception
        std::cerr << "Error during model inference:\n" << e.what() << std::endl;
        return;
    }


    // --- 5. Обработка результата ---
    // output_tensor содержит логарифмы вероятностей (log_softmax) для каждого из 3 классов
    // Нам нужен индекс класса с максимальным значением

    // argmax вдоль оси классов (ось 1)
    torch::Tensor predicted_idx_tensor = torch::argmax(output_tensor, 1); // Результат будет тензором формы [1]
    int64_t predicted_idx = predicted_idx_tensor.item<int64_t>(); // Извлекаем значение индекса

    // Получаем вероятности (опционально, если нужны именно они)
    torch::Tensor probabilities = torch::exp(output_tensor); // exp(log_softmax) -> softmax

    // Определяем имена классов (замените на ваши реальные имена)
    static std::vector<std::string> class_names = { "noise", "jarvis", "turnon", "codered", "wakeup" };

    // Получаем вероятность предсказанного класса
    float confidence = probabilities[0][predicted_idx].item<float>();

    // Порог вероятности
    const float THRESHOLD = 0.8f;

    if (predicted_idx != 0/* && predicted_idx != 3 */ && confidence >= THRESHOLD)
    {
        std::cout << "--- Inference Results ---" << std::endl;
        std::cout << "Raw model output (log probabilities): " << output_tensor << std::endl;
        std::cout << "Probabilities: " << probabilities << std::endl; // Раскомментируйте, если нужны вероятности
        std::cout << "Predicted class index: " << predicted_idx << std::endl;
        std::cout << "Predicted class name: " << class_names[predicted_idx] << std::endl;

        mfcc_buffer.clear();

    }

}