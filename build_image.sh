tar cvf codes.tar .bazelrc WORKSPACE BUILD ./include/ ./bazel/ ./3rdparty/ ./scripts/ ./tools/ ./epinions/
docker build -t frecsys_box .
