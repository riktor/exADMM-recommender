

def frecsys_library(name):
    native.cc_library(
        name = name,
        # These headers are made available to other targets.
        hdrs =
            native.glob(["include/frecsys/*.hpp"]),
        includes = [
            "include",
            "include/frecsys/*.hpp",
        ],
        visibility = ["//visibility:public"],
        deps = [
            "@com_gitlab_libeigen_eigen//:eigen",
            "@com_github_google_glog//:glog",
        ]
    )
