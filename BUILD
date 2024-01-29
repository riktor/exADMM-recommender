load("//:bazel/frecsys.bzl", "frecsys_library")
frecsys_library("frecsys")

TEST_DEPS = [
    "@frecsys",
    "@com_gitlab_libeigen_eigen//:eigen",
    "@com_github_google_glog//:glog",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@fmt",
]

DEPS = [
    "@frecsys",
    "@com_gitlab_libeigen_eigen//:eigen",
    "@com_github_google_glog//:glog",
    "@fmt",
]

EXECUTABLE_DEPS = [
    "@frecsys",
    "@com_gitlab_libeigen_eigen//:eigen",
    "@com_github_google_glog//:glog",
    "@fmt",
]

TEST_COPTS = [
    "-g",
    "-Wall",
    "-O0",
    "-march=native",
    "-std=c++2a",
    "-mavx",
    "-mavx2",
]

COPTS = [
    "-O3",
    "-march=native",
    "-std=c++2a",
    "-mavx",
    "-mavx2",
]

TEST_DATA = [
    "tests/ml-1m/train.csv",
    "tests/ml-1m/validation_tr.csv",
    "tests/ml-1m/validation_te.csv"
]

LINKOPTS = ["-lpthread"]

[cc_test(
    name = test_filename.split("/")[-1].split('.')[0],
    timeout = "moderate",
    srcs = [test_filename],
    copts = TEST_COPTS,

    # This is the data set that is bundled for the testing.
    data = TEST_DATA,
    deps = TEST_DEPS,
    linkopts = LINKOPTS,
) for test_filename in glob([
    "tests/*_test.cpp",
])]

cc_binary(
    name = "run_model",
    srcs = ["tools/run_model.cpp", "tools/CLI11/CLI11.hpp"],
    copts = COPTS,
    deps = EXECUTABLE_DEPS,
    linkopts = LINKOPTS,
)
