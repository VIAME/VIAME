
# clean build env
docker stop viame_installer_zip || true && docker rm --force viame_installer_zip || true
rm -rf viame-src-clone || true

git clone https://github.com/VIAME/VIAME.git viame-src-clone
cd viame-src-clone
git checkout next

# stand up a new docker build env
docker pull nvidia/cuda:12.3.2-cudnn9-devel-centos7
chmod +x cmake/build_server_centos.sh
docker run -td --runtime=nvidia --name viame_installer_zip nvidia/cuda:12.3.2-cudnn9-devel-centos7 bash
cd ../
docker cp viame-src-clone viame_installer_zip:/viame/

# run the build script in the fresh docker environment
docker exec -i viame_installer_zip ./viame/cmake/build_server_centos.sh
