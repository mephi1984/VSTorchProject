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

CnnNet model;
torch::Device device(torch::kCPU);

/*
int predict(std::vector<std::vector<float>>& mfcc_vector, int n_mfcc_out)
{
    std::string model_path = "C:\\Users\\User\\source\\repos\\TorchProject03\\TorchProject03\\model_mfcc2_3classess_saved.pt";

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

}*/


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


    mfcc_buffer.resize(T_FIXED, std::vector<float>(n_mfcc_out, 0.0f));
    

    std::string model_path = "C:\\Users\\User\\source\\repos\\TorchProject03\\TorchProject03\\model_mfcc2_3classes_tf100.pt";

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


    std::thread t(&threadFunction);

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

    t.join();

    return 0;
}

// Реализация функции processData (заглушка)
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

    mfcc_buffer_mutex.lock();

    
    if (mfcc_buffer.size() < T_FIXED-1)
    {
        mfcc_buffer.resize(T_FIXED-1, std::vector<float>(n_mfcc_out, 0.0f));
    }

    mfcc_buffer.insert(mfcc_buffer.end(), part_mfcc_vector[0]);


    while (mfcc_buffer.size() > T_FIXED)
    {
        mfcc_buffer.erase(mfcc_buffer.begin());
    }

    mfcc_buffer_mutex.unlock();
}

void threadFunction()
{
    int n_mfcc = 13;
    int num_frames = T_FIXED;


    while (!shouldExit)
    {
        ///static auto last_detection_time = std::chrono::steady_clock::now();
        //auto now = std::chrono::steady_clock::now();
        //std::chrono::duration<float> elapsed_seconds = now - last_detection_time;
        //static const float cooldown_seconds = 1.0f; // Например, 1 секунда


        mfcc_buffer_mutex.lock();


        std::array<float, 13 * 100> flat_mfcc;

        auto itr = mfcc_buffer.begin();
        for (size_t i = 0; i < 100; i++)
        {
            std::copy(itr->begin(), itr->end(), flat_mfcc.begin() + (i * 13));
            itr++;
        }
            /*
        static std::vector<float> flat_mfcc;
        flat_mfcc.clear();
        flat_mfcc.reserve(num_frames * n_mfcc);
        for (auto itr = mfcc_buffer.begin(); itr != mfcc_buffer.end(); itr++)
        {
            flat_mfcc.insert(flat_mfcc.end(), itr->begin(), itr->end());

        }*/
        mfcc_buffer_mutex.unlock();

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
        static std::vector<std::string> class_names = { "noise", "jarvis", "turnon" };

        if (predicted_idx != 0)
        {
            std::cout << "--- Inference Results ---" << std::endl;
            std::cout << "Raw model output (log probabilities): " << output_tensor << std::endl;
            std::cout << "Probabilities: " << probabilities << std::endl; // Раскомментируйте, если нужны вероятности
            std::cout << "Predicted class index: " << predicted_idx << std::endl;
            std::cout << "Predicted class name: " << class_names[predicted_idx] << std::endl;

        }
        //std::cout << "Probabilities: " << probabilities << std::endl; // Раскомментируйте, если нужны вероятности
        //std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Небольшая задержка в цикле
    }

}

/*
int main_old(int argc, char* argv[])
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

    std::vector<std::vector<float>> chunks;

    int shift = 0;

    int sampleCount = 400;
    int sampleStep = 160;
    while (shift + sampleStep + sampleCount < x.size())
    {
        std::vector<float> chunk(sampleCount);
        std::copy(x.begin()+shift, x.begin()+ shift + sampleCount, chunk.begin());
        chunks.push_back(chunk);
        shift += sampleStep;

    }


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

    std::vector<std::vector<float>> mfcc_vector;

    for (int i = 0; i < chunks.size(); i++)
    {
        auto part_mfcc_vector = librosa::Feature::mfcc(chunks[i], sr, n_fft, n_hop, "hann", false, "reflect", power, n_mel, fmin, fmax, n_mfcc_out, use_norm, dct_type);

        if (part_mfcc_vector.size() != 1)
        {
            return 0;
        }

        mfcc_vector.push_back(part_mfcc_vector[0]);
    }

    // Весь остальной код я поместил в отдельную функцию
    return predict(mfcc_vector, n_mfcc_out);
}*/