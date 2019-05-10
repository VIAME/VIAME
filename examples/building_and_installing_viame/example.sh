
mkdir -p ~/Dev/viame

git clone https://github.com/VIAME/VIAME.git ~/Dev/viame/src

cd ~/Dev/viame/src
git submodule update --init --recursive

cd ..
mkdir build

cmake ../src

make -j10
