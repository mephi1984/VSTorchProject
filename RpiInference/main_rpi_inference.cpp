#define _CRT_SECURE_NO_WARNINGS
#include "../wavreader.h"
#include <librosa/librosa.h> 

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <filesystem> 

#include <torch/torch.h>
#include <torch/script.h> 

#include <portaudio.h>

#include <list>

#include <thread>
#include <cstdlib>
#include <ao/ao.h>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <sndfile.h>
#include <memory>

// PA_ALSA_PLUGHW="1"


class AudioPlayer {
    public:
        AudioPlayer() : stopFlag(false) {
            workerThread = std::thread(&AudioPlayer::playerThread, this);
        }
    
        ~AudioPlayer() {
            stop();
            if (workerThread.joinable()) {
                workerThread.join();
            }
        }
   
        void enqueue(const std::string& filename) {
            std::lock_guard<std::mutex> lock(queueMutex);
            trackQueue.push(filename);
            queueCond.notify_one();  
        }
    

        void stop() {
            stopFlag = true;
            std::lock_guard<std::mutex> lock(queueMutex);
            while (!trackQueue.empty()) {
                trackQueue.pop();
            }
            queueCond.notify_one(); 
        }
    
        void clearQueue() {
            std::lock_guard<std::mutex> lock(queueMutex);
            while (!trackQueue.empty()) {
                trackQueue.pop();
            }
        }
    

        std::queue<std::string> trackQueue;  
        std::mutex queueMutex;  
        std::condition_variable queueCond;  
        std::atomic<bool> stopFlag;    
        std::thread workerThread;      

        volatile bool isPlaying = false;
    
        void playerThread() {
            while (!stopFlag) {
                std::string filename;
    
                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    queueCond.wait(lock, [this] {
                        return !trackQueue.empty() || stopFlag;
                    });
    
                    if (stopFlag) break;
                    isPlaying = true;
                    filename = trackQueue.front();
                    trackQueue.pop();
                }
    
                playWavFile(filename);
                isPlaying = false;
            }
        }
    

        void playWavFile(const std::string& filename) {


            SF_INFO sfinfo;
            SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfinfo);
            if (!file) {
                std::cerr << "Failed to open file: " << sf_strerror(NULL) << std::endl;
                Pa_Terminate();
                return;
            }

            std::cout << "File opened. Channels: " << sfinfo.channels 
                    << ", Sample rate: " << sfinfo.samplerate << std::endl;

            PaStream* stream;
            PaError paErr;
            paErr = Pa_OpenDefaultStream(
                &stream,
                0,                
                sfinfo.channels, 
                paFloat32,         
                sfinfo.samplerate,
                256,           
                nullptr,         
                nullptr         
            );

            if (paErr != paNoError) {
                std::cerr << "PortAudio stream error: " << Pa_GetErrorText(paErr) << std::endl;
                sf_close(file);
                Pa_Terminate();
                return;
            }

            paErr = Pa_StartStream(stream);
            if (paErr != paNoError) {
                std::cerr << "PortAudio start error: " << Pa_GetErrorText(paErr) << std::endl;
                Pa_CloseStream(stream);
                sf_close(file);
                Pa_Terminate();
                return;
            }

            constexpr int bufferSize = 1024;
            float buffer[bufferSize * sfinfo.channels]; 

            while (true) {
                sf_count_t framesRead = sf_readf_float(file, buffer, bufferSize);
                if (framesRead <= 0) break;

                paErr = Pa_WriteStream(stream, buffer, framesRead);
                if (paErr != paNoError) {
                    std::cerr << "PortAudio write error: " << Pa_GetErrorText(paErr) << std::endl;
                    break;
                }
            }

            Pa_StopStream(stream);
            Pa_CloseStream(stream);
            sf_close(file);
        }
    };
    
void processData(const std::vector<float>& data);

struct AudioData {
    std::vector<float> buffer;
    int samplesNeeded = 160;

};
using namespace std;


const int MFCC_DIM = 13;
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

CnnNet model;
torch::Device device(torch::kCPU);



static int audioCallback(const void* inputBuffer, void* outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags statusFlags,
    void* userData) {
    AudioData* data = static_cast<AudioData*>(userData);
    const int16_t* samples = static_cast<const int16_t*>(inputBuffer);


    (void)outputBuffer;

    for (unsigned long i = 0; i < framesPerBuffer; ++i) {
        float sample = static_cast<float>(samples[i]) / 32768.0f;
        data->buffer.push_back(sample);

        if (data->buffer.size() >= data->samplesNeeded) {
            processData(data->buffer);
            data->buffer.clear();
        }
    }

    return paContinue;
}

std::list<std::vector<float>> mfcc_buffer;

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

    selectedDevice = 1;

    return selectedDevice;
}

std::shared_ptr<AudioPlayer> audioPlayer;
AudioData audioData;

PaStream* stream;

int main() {
    int n_mfcc_out = 13;

    audioPlayer = std::make_shared<AudioPlayer>();

    std::string model_path = "/home/mephi/newmodel_pi_5classes_tf_mega_noaug200.pt";

    try {
        torch::load(model, model_path);
        model->eval();
        std::cout << "Model loaded successfully using torch::load from: " << model_path << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model with torch::load:\n" << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading the model with torch::load:\n" << e.what() << std::endl;
        return -1;
    }

    model->to(device);

    
    PaError err;

    err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return 1;
    }

    PaDeviceIndex inputDevice = selectInputDevice();
    if (inputDevice == paNoDevice) {
        Pa_Terminate();
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    audioPlayer->enqueue("/home/mephi/Media/351556__richerlandtv__connect3.wav");

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    

    PaStreamParameters inputParameters;
    inputParameters.device = inputDevice;
    inputParameters.channelCount = 1; 
    inputParameters.sampleFormat = paInt16; 
    inputParameters.suggestedLatency =
        Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;


    err = Pa_OpenStream(&stream,
        &inputParameters,
        nullptr,   
        16000,      
        160,      
        paClipOff, 
        audioCallback,
        &audioData);

    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        Pa_Terminate();
        return 1;
    }

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    std::cout << "Recording from "
        << Pa_GetDeviceInfo(inputParameters.device)->name
        << "\nPress Enter to stop..." << std::endl;

    int count = 0;
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        count ++;

        if (count > 4)
        {
            if (audioPlayer->trackQueue.size() != 0 || audioPlayer->isPlaying)
            {
                continue;
            }
            std::cout << "Restarting stream..." << std::endl;
            count = 0;
        

            err = Pa_StopStream(stream);
            if (err != paNoError) {
                std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
            }
        
            Pa_CloseStream(stream);

            audioPlayer->enqueue("/home/mephi/Media/351556__richerlandtv__connect3.wav");

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    

            mfcc_buffer.clear();

            err = Pa_OpenStream(&stream,
                &inputParameters,
                nullptr,   
                16000,      
                160,      
                paClipOff, 
                audioCallback,
                &audioData);

            err = Pa_StartStream(stream);
        }

    }

    err = Pa_StopStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
    }

    Pa_CloseStream(stream);
    Pa_Terminate();
    
    return 0;
}


void processData(const std::vector<float>& data) {

    int sr = 16000;
    int n_fft = 400;
    int n_hop = 160;
    int n_mel = 26;
    int n_mfcc_out = 13; 
    int fmin = 0;
    int fmax = 8000; 
    float power = 2.f;
    int dct_type = 2;
    bool use_norm = true; 

    if (audioPlayer->trackQueue.size() != 0 || audioPlayer->isPlaying)
    {
        return;
    }


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

    mfcc_buffer.insert(mfcc_buffer.end(), part_mfcc_vector[0]);

    if (mfcc_buffer.size() < T_FIXED)
    {
        return;
    }

    while (mfcc_buffer.size() > T_FIXED)
    {
        mfcc_buffer.erase(mfcc_buffer.begin());
    }

    int n_mfcc = 13;
    int num_frames = T_FIXED;

    static std::array<float, 13> silence = {
        -509.902, 3.32611e-05, -3.51126e-05, -2.25488e-05
    , -7.8028e-06, 1.38202e-05, 4.75442e-05
    ,-1.78539e-06, -5.57437e-05, -4.80071e-05, 7.07542e-06,
    -5.07843e-05, -7.4325e-05 };



    std::array<float, 13 * T_FIXED> flat_mfcc;

    auto itr = mfcc_buffer.begin();
    for (size_t i = 0; i < T_FIXED; i++)
    {
        if (i >= 1 && i <= 11)
        {
            std::copy(silence.begin(), silence.end(), flat_mfcc.begin() + (i * 13));
        }
        else
        {
            std::copy(itr->begin(), itr->end(), flat_mfcc.begin() + (i * 13));
        }
        itr++;
    }


    torch::Tensor input_tensor = torch::from_blob(flat_mfcc.data(), { num_frames, n_mfcc }, torch::kFloat);

    input_tensor = input_tensor.clone();

    input_tensor = input_tensor.transpose(0, 1);

    input_tensor = input_tensor.unsqueeze(0); 

    input_tensor = input_tensor.to(device);


    torch::Tensor output_tensor;
    try {
        torch::NoGradGuard no_grad;
        output_tensor = model->forward(input_tensor);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error during model inference:\n" << e.what() << std::endl;
        return;
    }
    catch (const std::exception& e) { 
        std::cerr << "Error during model inference:\n" << e.what() << std::endl;
        return;
    }


    torch::Tensor predicted_idx_tensor = torch::argmax(output_tensor, 1);
    int64_t predicted_idx = predicted_idx_tensor.item<int64_t>(); 

    torch::Tensor probabilities = torch::exp(output_tensor);

    static std::vector<std::string> class_names = { "noise", "jarvis", "turnon", "codered", "wakeup" };

    float confidence = probabilities[0][predicted_idx].item<float>();

    std::cout << "-Probabilities: " << probabilities << std::endl;
      
    //if (predicted_idx != 0 && predicted_idx !=3 && (predicted_idx == 2 && confidence >= 0.6 || predicted_idx == 4 && confidence >= 0.4 || (predicted_idx == 1 && confidence >= 0.3)))
    if (predicted_idx != 0 && predicted_idx !=3 && (predicted_idx == 2 && confidence >= 0.6 || predicted_idx == 4 && confidence >= 0.6 || (predicted_idx == 1 && confidence >= 0.5)))
    {
        std::cout << "--- Inference Results ---" << std::endl;
        std::cout << "Raw model output (log probabilities): " << output_tensor << std::endl;
        std::cout << "Probabilities: " << probabilities << std::endl;
        std::cout << "Predicted class index: " << predicted_idx << std::endl;
        std::cout << "Predicted class name: " << class_names[predicted_idx] << std::endl;

        mfcc_buffer.clear();

        if (predicted_idx == 1)
        {
            audioPlayer->enqueue("/home/mephi/Media/reply_jarvis.wav");
        }
        else if (predicted_idx == 2)
        {
            audioPlayer->enqueue("/home/mephi/Media/build_reply.wav");
        }
        else if (predicted_idx == 4)
        {
            audioPlayer->enqueue("/home/mephi/Media/reply_wakeup.wav");
        }


    }

}