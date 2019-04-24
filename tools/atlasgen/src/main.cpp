/*
 * Copyright (C) 2019 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gltfio/MeshPipeline.h>

#include <utils/Path.h>

#include <getopt/getopt.h>

#include <fstream>
#include <iostream>
#include <string>

static uint8_t g_flattenFlags = gltfio::MeshPipeline::FILTER_TRIANGLES;
static bool g_flattenOnly = true;

static const char* USAGE = R"TXT(
ATLASGEN consumes a glTF 2.0 file and produces a new glTF file that adds a new UV set to each mesh
suitable for baking lightmaps. However this does much more than add a UV set; some of the changes
in the output asset include:

    - Vertices might be inserted into the geometry.
    - Shared meshes might become duplicated.
    - Transforms get baked into the position data.
    - Position attributes all become XYZ fp32.
    - Primitives with mode != TRIANGLES are removed.

Usage:
    ATLASGEN [options] <input path> <output filename> ...

Options:
   --help, -h
       Print this message
   --license, -L
       Print copyright and license information
   --discard, -d
       Discard all textures from the original model

Example:
    ATLASGEN -d bistro_in.gltf bistro_out.gltf
)TXT";

static void printUsage(const char* name) {
    std::string execName(utils::Path(name).getName());
    const std::string from("ATLASGEN");
    std::string usage(USAGE);
    for (size_t pos = usage.find(from); pos != std::string::npos; pos = usage.find(from, pos)) {
        usage.replace(pos, from.length(), execName);
    }
    puts(usage.c_str());
}

static void license() {
    std::cout <<
    #include "licenses/licenses.inc"
    ;
}

static int handleArguments(int argc, char* argv[]) {
    static constexpr const char* OPTSTR = "hLd";
    static const struct option OPTIONS[] = {
        { "help",     no_argument, 0, 'h' },
        { "license",  no_argument, 0, 'L' },
        { "discard",  no_argument, 0, 'd' },
        { }
    };

    int opt;
    int optionIndex = 0;

    while ((opt = getopt_long(argc, argv, OPTSTR, OPTIONS, &optionIndex)) >= 0) {
        std::string arg(optarg ? optarg : "");
        switch (opt) {
            default:
            case 'h':
                printUsage(argv[0]);
                exit(0);
            case 'L':
                license();
                exit(0);
            case 'd':
                g_flattenFlags |= gltfio::MeshPipeline::DISCARD_TEXTURES;
        }
    }

    return optind;
}

static std::ifstream::pos_type getFileSize(const char* filename) {
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

int main(int argc, char* argv[]) {
    const int optionIndex = handleArguments(argc, argv);
    const int numArgs = argc - optionIndex;
    if (numArgs < 2) {
        printUsage(argv[0]);
        return 1;
    }

    utils::Path inputPath = argv[optionIndex];
    if (!inputPath.exists()) {
        std::cerr << inputPath << " not found!" << std::endl;
        return 1;
    }
    if (inputPath.isDirectory()) {
        auto files = inputPath.listContents();
        for (auto file : files) {
            if (file.getExtension() == "gltf") {
                inputPath = file;
                std::cout << "Found " << inputPath.getName() << std::endl;
                break;
            }
        }
        if (inputPath.isDirectory()) {
            std::cerr << "no glTF file found in " << inputPath << std::endl;
            return 1;
        }
    }

    utils::Path outputPath = argv[optionIndex + 1];
    if (inputPath.getExtension() != "gltf" || outputPath.getExtension() != "gltf") {
        std::cerr << "File extension must be gltf." << std::endl;
        return 1;
    }

    // Import the glTF file.
    gltfio::MeshPipeline pipeline;
    gltfio::MeshPipeline::AssetHandle asset = pipeline.load(inputPath);
    if (!asset) {
        std::cerr << "Unable to read " << inputPath << std::endl;
        exit(1);
    }

    // Flatten the mesh structure: bake out transforms, etc.
    asset = pipeline.flatten(asset, g_flattenFlags);

    // Generate atlases.
    #if 0
    if (!g_flattenOnly) {
        asset = pipeline.parameterize(asset);
    }
    #endif

    // Export the JSON and BIN files.
    std::string binFilename = outputPath.getNameWithoutExtension() + ".bin";
    utils::Path binFullpath = outputPath.getParent() + binFilename;
    pipeline.save(asset, outputPath, binFullpath);
    std::cout << "Generated " << outputPath << " and " << binFullpath << std::endl;
}
