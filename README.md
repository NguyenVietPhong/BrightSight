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

# install tritionclient #
  pip3 install tritonclient[all]


# Chạy chương trình
  python3 run.py
