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
#include <fstream>
#include <portaudio.h>

#include <list>
#include "sndfile.h"

namespace fs = std::filesystem;

static bool timeToExit = false;

std::string find_next_available_filename(const std::string& suffix = ".csv") {
    const std::string prefix = "wakeup";
    int next_number = 1;
    const int max_files = 10000; // ������������ ���������� ������ ��� ��������

    for (; next_number <= max_files; ++next_number) {
        // ����������� ����� � �������� ������ (4 �����)
        std::ostringstream oss;
        oss << prefix << std::setw(4) << std::setfill('0') << next_number << suffix;
        std::string filename = oss.str();

        // ���������, ���������� �� ����
        if (!fs::exists(filename)) {
            return filename;
        }
    }

    // ���� ��� ����� �� max_files ����������, ���������� ������ ������
    return "";
}

// PA_ALSA_PLUGHW="1"


struct AudioData {
    std::vector<float> buffer;
    int samplesNeeded = 160;

    SNDFILE* wavFile = nullptr;
    SF_INFO sfInfo;
    
    AudioData() {
        sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
        sfInfo.channels = 1;
        sfInfo.samplerate = 16000;

        std::string fileName = find_next_available_filename(".wav");

        
        wavFile = sf_open(fileName.c_str(), SFM_WRITE, &sfInfo);
        if (!wavFile) {
            std::cerr << "Error opening WAV file: " << sf_strerror(nullptr) << std::endl;
        }
    }
    
    ~AudioData() {
        if (wavFile) {
            sf_close(wavFile);
        }
    }

    // � ��������� AudioData �������� �����:
    void finishRecording() {
        if (wavFile) {
            sf_close(wavFile);
            wavFile = nullptr; // �������� ��� ��������
        }
    }

};

void processData(const std::vector<float>& data, AudioData* audioData );

using namespace std;


const int MFCC_DIM = 13;
const int64_t MAX_NEGATIVES = 500;


constexpr int T_FIXED = 140;


static int audioCallback(const void* inputBuffer, void* outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags statusFlags,
    void* userData) {

        if (timeToExit)
        {
            return paContinue;
        }

    
    AudioData* data = static_cast<AudioData*>(userData);
    const int16_t* samples = static_cast<const int16_t*>(inputBuffer);
    (void)outputBuffer;

    if (data->wavFile) {
        sf_write_short(data->wavFile, samples, framesPerBuffer);
    }

    for (unsigned long i = 0; i < framesPerBuffer; ++i) {
        float sample = static_cast<float>(samples[i]) / 32768.0f;
        data->buffer.push_back(sample);

        if (data->buffer.size() >= data->samplesNeeded) {
            processData(data->buffer, data);
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
    /*
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
    }*/

    return selectedDevice;
}


int main() {


    PaError err;
    AudioData audioData;

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

    PaStreamParameters inputParameters;
    inputParameters.device = inputDevice;
    inputParameters.channelCount = 1; 
    inputParameters.sampleFormat = paInt16; 
    inputParameters.suggestedLatency =
        Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;

    PaStream* stream;

    try{

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
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
    }
    catch (...)
    {
        std::cout << "time to stop" << std::endl;
    }

    //audioData.finishRecording();

    err = Pa_StopStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
    }

    Pa_CloseStream(stream);
    Pa_Terminate();
    
    return 0;
}


void processData(const std::vector<float>& data, AudioData* audioData ) {

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
    std::string csvFileName = find_next_available_filename(".csv");

    std::ofstream fout(csvFileName);


    for (auto itr = mfcc_buffer.begin(); itr != mfcc_buffer.end(); itr++) {

        fout << ((*itr)[0]);
        for (int j = 1; j < (*itr).size(); j++)
        {
            fout << ";" << ((*itr)[j]);
        }
        fout << std::endl;
    }
    fout.close();


    timeToExit = true;
    audioData->finishRecording();
    throw std::runtime_error("Time to exit");
}