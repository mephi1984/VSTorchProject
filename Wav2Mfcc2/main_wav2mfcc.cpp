#define _CRT_SECURE_NO_WARNINGS
#include "../wavreader.h"
#include <librosa/librosa.h>

#include <iostream>
#include <vector>

#include <chrono>
#include <numeric>
#include <algorithm>
#include <fstream>

using namespace std;

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cout << "Usage: wav2mfcc2.exe /path/to/file.wav /output/path/to/file.csv" << std::endl;
        return -1;
    }
    std::string wavFileName = argv[1];
    void* h_x = wav_read_open(wavFileName.c_str());
    //void* h_x = wav_read_open("D:\\Work\\media\\jarvis16000\\0000_16k_mono.wav");

    int format, channels, sr, bits_per_sample;
    unsigned int data_length;
    int res = wav_get_header(h_x, &format, &channels, &sr, &bits_per_sample, &data_length);
    if (!res)
    {
        cerr << "get ref header error: " << res << endl;
        return -1;
    }

    int samples = data_length * 8 / bits_per_sample;
    std::vector<int16_t> tmp(samples);
    res = wav_read_data(h_x, reinterpret_cast<unsigned char*>(tmp.data()), data_length);
    if (res < 0)
    {
        cerr << "read wav file error: " << res << endl;
        return -1;
    }
    std::vector<float> x(samples);
    std::transform(tmp.begin(), tmp.end(), x.begin(),
        [](int16_t a) {
            return static_cast<float>(a) / 32767.f;
        });

    std::cout << "Sample rate: " << sr << "Hz" << std::endl;

    int n_fft = 400;
    int n_hop = 160;
    int n_mel = 26;
    int fmin = 0;
    int fmax = 8000;
    float power = 2.f;

    auto melspectrogram_start_time = std::chrono::system_clock::now();
    std::vector<std::vector<float>> mels = librosa::Feature::melspectrogram(x, sr, n_fft, n_hop, "hann", false, "reflect", power, n_mel, fmin, fmax);
    auto melspectrogram_end_time = std::chrono::system_clock::now();
    auto melspectrogram_duration = std::chrono::duration_cast<std::chrono::milliseconds>(melspectrogram_end_time - melspectrogram_start_time);
    std::cout << "Melspectrogram runing time is " << melspectrogram_duration.count() << "ms" << std::endl;

    assert(!mels.empty());
    std::cout << "Verify the energy of melspectrogram features:" << std::endl;
    std::cout << "mel.dims: [" << mels.size() << "," << mels[0].size() << "]" << std::endl;

    auto mfcc_vector = librosa::Feature::mfcc(x, sr, n_fft, n_hop, "hann", false, "reflect", power, n_mel, fmin, fmax, 13, true, 2);
    int count = mfcc_vector.size();

    std::string csvFileName = argv[2];
    std::ofstream fout(csvFileName);

    for (int i = 0; i < mfcc_vector.size(); i++) {

        fout << (mfcc_vector[i][0]);
        for (int j = 1; j < mfcc_vector[i].size(); j++)
        {
            fout << ";" << (mfcc_vector[i][j]);
        }
        fout << std::endl;
    }
    fout.close();


    /*
    std::cout << count << std::endl;
    std::cout << std::endl;
    for (int i = 0; i < mfcc_vector.size(); i++) {

        for (int j = 0; j < mfcc_vector[i].size(); j++)
        {
            std::cout << (mfcc_vector[i][j]) << ";";
        }
        std::cout << std::endl;
    }
    */


    return 0;
}