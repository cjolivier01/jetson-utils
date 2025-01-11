config_setting(
    name = "aarch64-linux-gnu",
    constraint_values = ["@platforms//cpu:aarch64"],
)

config_setting(
    name = "x86_64-linux-gnu",
    constraint_values = ["@platforms//cpu:x86_64"],
)

cc_library(
    name = "libsoup",
    hdrs = glob([
        "include/libsoup-2.4/**/*.h*",
        "include/nlohmann/**/*.h*",
    ]),
    includes = [
        "include/libsoup-2.4",
        "include/nlohmann",
    ],
    linkopts = select({
        ":aarch64-linux-gnu": [
            "-l:/usr/lib/aarch64-linux-gnu/libsoup-2.4.so",
        ],
        ":x86_64-linux-gnu": ["-l:/usr/lib/x86_64-linux-gnu/libsoup-2.4.so"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "@glib",
        "@json_glib",
    ],
)
