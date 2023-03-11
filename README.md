# BrightSight
# Cài đặt môi trường trên Jetson Nano 
  sudo apt-get install virtualenv
  
  python3 -m virtualenv -p python3 env
  
  source env/bin/activate

#  update and  upgrade
  sudo apt-get -y update
  
  sudo apt-get -y upgrade

# Dependencies
  sudo apt-get install python3-setuptools

# install speech_text
  sudo apt-get install portaudio19-dev -y
  
  pip3 install -r speech_text/requirements.txt
  
  sudo apt-get install flac

# install image captioning
# install pytorch 
# install the dependencies (if not already onboard)

  sudo apt-get install python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
  
  sudo -H pip3 install future
  
  sudo pip3 install -U --user wheel mock pillow
  
  sudo -H pip3 install testresources
  
# above 58.3.0 you get version issues

  sudo -H pip3 install setuptools==58.3.0
  
  sudo -H pip3 install Cython
  
# install gdown to download from Google drive

  sudo -H pip3 install gdown
  
# download the wheel
  gdown https://drive.google.com/uc?id=1TqC6_2cwqiYacjoLhLgrZoap6-sVL2sd
  
# install PyTorch 1.10.0
  sudo -H pip3 install torch-1.10.0a0+git36449ea-cp36-cp36m-linux_aarch64.whl
# clean up
  rm torch-1.10.0a0+git36449ea-cp36-cp36m-linux_aarch64.whl


# install torchvison
# Used with PyTorch 1.10.0
# the dependencies
  sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev
  
  sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
  
  sudo pip3 install -U pillow
  
# install gdown to download from Google drive, if not done yet
  sudo -H pip3 install gdown
# download TorchVision 0.11.0
  gdown https://drive.google.com/uc?id=1C7y6VSIBkmL2RQnVy8xF9cAnrrpJiJ-K
# install TorchVision 0.11.0
  sudo -H pip3 install torchvision-0.11.0a0+fa347eb-cp36-cp36m-linux_aarch64.whl
# clean up
  rm torchvision-0.11.0a0+fa347eb-cp36-cp36m-linux_aarch64.whl

# Cài đặt face_recognition
  sudo apt-get update
  
  sudo apt-get install python3-pip cmake libopenblas-dev liblapack-dev libjpeg-dev
  
  # Trước khi tiếp tục, chúng ta cần tạo một tệp hoán đổi. Jetson Nano chỉ có 4GB RAM sẽ không đủ để biên dịch dlib. Để giải quyết vấn đề này, chúng tôi sẽ thiết lập   một tệp hoán đổi cho phép chúng tôi sử dụng dung lượng ổ đĩa làm RAM bổ sung.
  
  git clone https://github.com/JetsonHacksNano/installSwapfile
  
  ./installSwapfile/installSwapfile.sh
  
  # download dlib
  wget http://dlib.net/files/dlib-19.17.tar.bz2 
  
  tar jxvf dlib-19.17.tar.bz2
  
  cd dlib-19.17
  
  #Trước khi biên dịch dlib, chúng ta cần chú thích một dòng dòng 854

  gedit dlib/cuda/cudnn_dlibapi.cpp
  
  //forward_algo = forward_best_algo;
  
  #Tiếp theo, chạy các lệnh này để biên dịch và cài đặt dlib

  sudo python3 setup.py install
  
 #Cuối cùng, chúng ta cần cài đặt thư viện face_recognition

  sudo pip3 install face_recognition

# install tritionclient #
  pip3 install tritonclient[all]


# Chạy chương trình
  python3 run.py
