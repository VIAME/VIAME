
# Hack for certain versions of cudnn installs on some OS
if [ -f /usr/include/cudnn_v9.h ] && [ ! -f /usr/include/cudnn.h ]; then
 ln -s /usr/include/cudnn_v9.h /usr/include/cudnn.h
 ln -s /usr/include/cudnn_adv_v9.h /usr/include/cudnn_adv.h
 ln -s /usr/include/cudnn_cnn_v9.h /usr/include/cudnn_cnn.h
 ln -s /usr/include/cudnn_ops_v9.h /usr/include/cudnn_ops.h
 ln -s /usr/include/cudnn_version_v9.h /usr/include/cudnn_version.h
 ln -s /usr/include/cudnn_backend_v9.h /usr/include/cudnn_backend.h
 ln -s /usr/include/cudnn_graph_v9.h /usr/include/cudnn_graph.h
fi
