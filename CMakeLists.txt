cmake_minimum_required(VERSION 3.0)
project(test_libtorch)

#set(CMAKE_PREFIX_PATH "$ENV{LIBTORCH}")
set(CMAKE_PREFIX_PATH "/home/mephi/Work/portaudio/portaudio-19.7.0/build-rpi/install")
find_package(Torch REQUIRED)

find_library(AO_LIB ao)
find_library(VORBIS_LIB vorbis)
find_library(VORBISFILE_LIB vorbisfile)


add_executable(rpiInference wavreader.c RpiInference/main_rpi_inference.cpp)
target_link_libraries(rpiInference "${TORCH_LIBRARIES}")
target_include_directories(rpiInference PUBLIC /home/mephi/Work/portaudio/portaudio-19.7.0/build-rpi/install/include)
target_include_directories(rpiInference PUBLIC /home/mephi/Work/LibrosaCpp)
target_link_directories(rpiInference PUBLIC /home/mephi/Work/portaudio/portaudio-19.7.0/build-rpi/install/lib)
target_link_libraries(rpiInference portaudio sndfile)
#target_link_libraries(rpiInference samplerate)

target_link_libraries(rpiInference
    ${AO_LIB}
    ${VORBIS_LIB}
    ${VORBISFILE_LIB}
    pthread
)
set_property(TARGET rpiInference PROPERTY CXX_STANDARD 17)


add_executable(soundRecorderPi wavreader.c SoundRecorderPi/main_recorder.cpp)
target_include_directories(soundRecorderPi PUBLIC /home/mephi/Work/portaudio/portaudio-19.7.0/build-rpi/install/include)
target_include_directories(soundRecorderPi PUBLIC /home/mephi/Work/LibrosaCpp)
target_link_directories(soundRecorderPi PUBLIC /home/mephi/Work/portaudio/portaudio-19.7.0/build-rpi/install/lib)
target_link_libraries(soundRecorderPi portaudio sndfile)
set_property(TARGET soundRecorderPi PROPERTY CXX_STANDARD 17)
