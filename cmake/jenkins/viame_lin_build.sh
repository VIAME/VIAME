# clean docker build env
chmod +x viame_centos_upload.sh
mv viame_centos_upload.sh ../
cd ..
#docker run -td -v $(pwd)/VIAME_release_Centos:/viame/ --name viame_centos centos:7 bash || true
#docker exec -i viame_centos rm -rf /viame/ || true
#docker exec -i viame_centos rm -rf /viame/.* || true
docker stop viame_centos || true && docker rm --force viame_centos || true

rm -rf VIAME_release_Centos || true
git clone https://github.com/VIAME/VIAME.git VIAME_release_Centos
mv viame_centos_upload.sh VIAME_release_Centos
cd VIAME_release_Centos

# stand up a new docker build env
docker logout
docker pull nvidia/cuda:12.3.2-cudnn9-devel-centos7
chmod +x cmake/build_server_centos.sh
docker run -td --runtime=nvidia --name viame_centos nvidia/cuda:12.3.2-cudnn9-devel-centos7 bash
cd ../
docker cp VIAME_release_Centos viame_centos:/viame/
# run the script in the new docker build env
docker exec -i viame_centos ./viame/cmake/build_server_centos.sh
