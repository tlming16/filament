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

#include <xatlas.h>

#define CGLTF_WRITE_IMPLEMENTATION
#include "cgltf_write.h"

#include <utils/Log.h>

#include <math/mat4.h>
#include <math/vec3.h>
#include <math/vec4.h>

#include <fstream>
#include <vector>

using namespace filament::math;
using namespace utils;

using std::vector;

namespace {

using AssetHandle = gltfio::MeshPipeline::AssetHandle;

static const char* POSITION = "POSITION";
static const char* GENERATOR_ID = "gltfio";

// Bookkeeping structure for baking a single primitive + node pair.
struct BakedPrim {
    const cgltf_node* sourceNode;
    const cgltf_mesh* sourceMesh;
    const cgltf_primitive* sourcePrimitive;
    float3* bakedPositions;
    uint32_t* bakedIndices;
    float3 bakedMin;
    float3 bakedMax;
};

// Utility class to help populate cgltf arrays and ensure that the memory is freed.
template <typename T>
class ArrayHolder {
    vector<std::unique_ptr<T[]> > mItems;
public:
    T* alloc(size_t count) {
        mItems.push_back(std::make_unique<T[]>(count));
        return mItems.back().get();
    }
};

// Private implementation for MeshPipeline.
class Pipeline {
public:
    // Aggregate all buffers into a single buffer.
    const cgltf_data* flattenBuffers(const cgltf_data* sourceAsset);

    // Bake transforms and make each primitive correspond to a single node.
    const cgltf_data* flattenPrims(const cgltf_data* sourceAsset, uint32_t flags);

    // Use xatlas to generate a new UV set and modify topology appropriately.
    const cgltf_data* parameterize(const cgltf_data* sourceAsset);

    // Take ownership of the given asset and free it when the pipeline is destroyed.
    void addSourceAsset(cgltf_data* asset);

    ~Pipeline();

private:
    void bakeTransform(BakedPrim* prim, const mat4f& transform);
    void populateResult(const BakedPrim* prims, size_t numPrims, size_t numVerts);
    bool filterPrim(const cgltf_primitive& prim);

    uint32_t mFlattenFlags;
    vector<cgltf_data*> mSourceAssets;

    struct {
        ArrayHolder<cgltf_data> resultAssets;
        ArrayHolder<cgltf_scene> scenes;
        ArrayHolder<cgltf_node*> nodePointers;
        ArrayHolder<cgltf_node> nodes;
        ArrayHolder<cgltf_mesh> meshes;
        ArrayHolder<cgltf_primitive> prims;
        ArrayHolder<cgltf_attribute> attributes;
        ArrayHolder<cgltf_accessor> accessors;
        ArrayHolder<cgltf_buffer_view> views;
        ArrayHolder<cgltf_buffer> buffers;
        ArrayHolder<cgltf_image> images;
        ArrayHolder<cgltf_texture> textures;
        ArrayHolder<cgltf_material> materials;
        ArrayHolder<uint8_t> bufferData;
    } mStorage;
};

// Performs in-place mutation of a cgltf primitive to ensure that the POSITION attribute is the
// first item in its list of attributes, which makes life easier for pipeline operations.
void movePositionAttribute(cgltf_primitive* prim) {
    for (cgltf_size i = 0; i < prim->attributes_count; ++i) {
        if (prim->attributes[i].type == cgltf_attribute_type_position) {
            std::swap(prim->attributes[i], prim->attributes[0]);
            return;
        }
    }
}

// Returns true if the given cgltf asset has been flattened by the mesh pipeline and is therefore
// amenable to subsequent pipeline operations like baking and exporting.
bool isFlattened(const cgltf_data* asset) {
    return asset && asset->buffers_count == 1 && asset->nodes_count == asset->meshes_count &&
            asset->asset.generator == GENERATOR_ID;
}

std::ifstream::pos_type getFileSize(const char* filename) {
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

// Returns true if the given primitive should be baked out, false if it should be culled away.
bool Pipeline::filterPrim(const cgltf_primitive& prim) {
    const bool filterTriangles = mFlattenFlags & gltfio::MeshPipeline::FILTER_TRIANGLES;
    if (filterTriangles && prim.type != cgltf_primitive_type_triangles) {
        return false;
    }
    const cgltf_attribute* pos = prim.attributes_count ? prim.attributes : nullptr;
    if (!pos || !pos->data || !pos->data->count) {
        return false;
    }
    if (pos->data->is_sparse) {
        return false;
    }
    if (!prim.indices || prim.indices->is_sparse) {
        // TODO: generate trivial indices
        return false;
    }
    return true;
}

const cgltf_data* Pipeline::flattenBuffers(const cgltf_data* sourceAsset) {
    // Determine the total required size for the aggregated buffer.
    size_t totalSize = 0;
    for (size_t i = 0, len = sourceAsset->buffers_count; i < len; ++i) {
        totalSize += sourceAsset->buffers[i].size;
    }

    // Count the total number of attributes and primitives.
    size_t attribsCount = 0;
    size_t primsCount = 0;
    for (size_t i = 0, len = sourceAsset->meshes_count; i < len; ++i) {
        const auto& mesh = sourceAsset->meshes[i];
        primsCount += mesh.primitives_count;
        for (size_t j = 0; j < mesh.primitives_count; ++j) {
            attribsCount += mesh.primitives[j].attributes_count;
        }
    }

    // Count the total number of referenced nodes.
    size_t nodePointersCount = 0;
    for (size_t i = 0, len = sourceAsset->scenes_count; i < len; ++i) {
        const auto& scene = sourceAsset->scenes[i];
        nodePointersCount += scene.nodes_count;
    }

    // Allocate inner lists.
    uint8_t* bufferData = mStorage.bufferData.alloc(totalSize);
    cgltf_primitive* primitives = mStorage.prims.alloc(primsCount);
    cgltf_attribute* attributes = mStorage.attributes.alloc(attribsCount);
    cgltf_node** nodePointers = mStorage.nodePointers.alloc(nodePointersCount);

    // Allocate top-level structs.
    cgltf_buffer* buffer = mStorage.buffers.alloc(1);
    cgltf_buffer_view* views = mStorage.views.alloc(sourceAsset->buffer_views_count);
    cgltf_accessor* accessors = mStorage.accessors.alloc(sourceAsset->accessors_count);
    cgltf_image* images = mStorage.images.alloc(sourceAsset->images_count);
    cgltf_texture* textures = mStorage.textures.alloc(sourceAsset->textures_count);
    cgltf_material* materials = mStorage.materials.alloc(sourceAsset->materials_count);
    cgltf_mesh* meshes = mStorage.meshes.alloc(sourceAsset->meshes_count);
    cgltf_node* nodes = mStorage.nodes.alloc(sourceAsset->nodes_count);
    cgltf_scene* scenes = mStorage.scenes.alloc(sourceAsset->scenes_count);
    cgltf_data* resultAsset = mStorage.resultAssets.alloc(1);

    // Populate the new buffer object.
    size_t offset = 0;
    vector<size_t> offsets(sourceAsset->buffers_count);
    for (size_t i = 0, len = sourceAsset->buffers_count; i < len; ++i) {
        size_t size = sourceAsset->buffers[i].size;
        memcpy(bufferData + offset, sourceAsset->buffers[i].data, size);
        offsets[i] = offset;
        offset += size;
    }
    buffer->size = totalSize;
    buffer->data = bufferData;

    // Populate the buffer views.
    for (size_t i = 0, len = sourceAsset->buffer_views_count; i < len; ++i) {
        auto& view = views[i] = sourceAsset->buffer_views[i];
        size_t bufferIndex = view.buffer - sourceAsset->buffers;
        view.buffer = buffer;
        view.offset += offsets[bufferIndex];
    }

    // Clone the accessors.
    for (size_t i = 0, len = sourceAsset->accessors_count; i < len; ++i) {
        auto& accessor = accessors[i] = sourceAsset->accessors[i];
        accessor.buffer_view = views + (accessor.buffer_view - sourceAsset->buffer_views);
    }

    // Clone the images.
    for (size_t i = 0, len = sourceAsset->images_count; i < len; ++i) {
        auto& image = images[i] = sourceAsset->images[i];
        if (image.buffer_view) {
            image.buffer_view = views + (image.buffer_view - sourceAsset->buffer_views);
        }
    }

    // Clone the textures.
    for (size_t i = 0, len = sourceAsset->textures_count; i < len; ++i) {
        auto& texture = textures[i] = sourceAsset->textures[i];
        texture.image = images + (texture.image - sourceAsset->images);
    }

    // Clone the nodes.
    for (size_t i = 0, len = sourceAsset->nodes_count; i < len; ++i) {
        auto& node = nodes[i] = sourceAsset->nodes[i];
        if (node.mesh) {
            node.mesh = meshes + (node.mesh - sourceAsset->meshes);
        }
    }

    // Clone the scenes.
    for (size_t i = 0, len = sourceAsset->scenes_count; i < len; ++i) {
        const auto& sourceScene = sourceAsset->scenes[i];
        auto& resultScene = scenes[i] = sourceScene;
        resultScene.nodes = nodePointers;
        for (size_t j = 0; j < sourceScene.nodes_count; ++j) {
            resultScene.nodes[j] = nodes + (sourceScene.nodes[j] - sourceAsset->nodes);
        }
        nodePointers += sourceScene.nodes_count;
    }

    // Clone the materials.
    for (size_t i = 0, len = sourceAsset->materials_count; i < len; ++i) {
        auto& material = materials[i] = sourceAsset->materials[i];
        auto& t0 = material.pbr_metallic_roughness.base_color_texture.texture;
        auto& t1 = material.pbr_metallic_roughness.metallic_roughness_texture.texture;
        auto& t2 = material.pbr_specular_glossiness.diffuse_texture.texture;
        auto& t3 = material.pbr_specular_glossiness.specular_glossiness_texture.texture;
        auto& t4 = material.normal_texture.texture;
        auto& t5 = material.occlusion_texture.texture;
        auto& t6 = material.emissive_texture.texture;
        t0 = textures + (t0 - sourceAsset->textures);
        t1 = textures + (t1 - sourceAsset->textures);
        t2 = textures + (t2 - sourceAsset->textures);
        t3 = textures + (t3 - sourceAsset->textures);
        t4 = textures + (t4 - sourceAsset->textures);
        t5 = textures + (t5 - sourceAsset->textures);
        t6 = textures + (t6 - sourceAsset->textures);
    }

    // Clone the meshes, primitives, and attributes.
    for (size_t i = 0, len = sourceAsset->meshes_count; i < len; ++i) {
        const auto& sourceMesh = sourceAsset->meshes[i];
        auto& resultMesh = meshes[i] = sourceMesh;
        resultMesh.primitives = primitives;
        for (size_t j = 0; j < sourceMesh.primitives_count; ++j) {
            const auto& sourcePrim = sourceMesh.primitives[j];
            auto& resultPrim = resultMesh.primitives[j] = sourcePrim;
            resultPrim.material = materials + (sourcePrim.material - sourceAsset->materials);
            resultPrim.attributes = attributes;
            resultPrim.indices = accessors + (sourcePrim.indices - sourceAsset->accessors);
            for (size_t k = 0; k < sourcePrim.attributes_count; ++k) {
                const auto& sourceAttr = sourcePrim.attributes[k];
                auto& resultAttr = resultPrim.attributes[k] = sourceAttr;
                resultAttr.data = accessors + (sourceAttr.data - sourceAsset->accessors);
            }
            attributes += sourcePrim.attributes_count;
        }
        primitives += sourceMesh.primitives_count;
    }

    // Clone the high-level asset structure, then substitute some of the top-level lists.
    *resultAsset = *sourceAsset;
    resultAsset->buffers = buffer;
    resultAsset->buffers_count = 1;
    resultAsset->buffer_views = views;
    resultAsset->accessors = accessors;
    resultAsset->images = images;
    resultAsset->textures = textures;
    resultAsset->materials = materials;
    resultAsset->meshes = meshes;
    resultAsset->nodes = nodes;
    resultAsset->scenes = scenes;
    resultAsset->scene = scenes + (sourceAsset->scene - sourceAsset->scenes);
    return resultAsset;
}

const cgltf_data* Pipeline::flattenPrims(const cgltf_data* sourceAsset, uint32_t flags) {
    mFlattenFlags = flags;

    // This must be called after flattenBuffers.
    assert(sourceAsset->buffers_count == 1);

    // Sanitize each attribute list such that POSITIONS is always the first entry.
    // Also determine the number of primitives that will be baked.
    size_t numPrims = 0;
    size_t numAttributes;
    for (cgltf_size i = 0; i < sourceAsset->nodes_count; ++i) {
        const cgltf_node& node = sourceAsset->nodes[i];
        if (node.mesh) {
            for (cgltf_size j = 0; j < node.mesh->primitives_count; ++j) {
                cgltf_primitive& sourcePrim = node.mesh->primitives[j];
                movePositionAttribute(&sourcePrim);
                if (filterPrim(sourcePrim)) {
                    numPrims++;
                    numAttributes += sourcePrim.attributes_count;
                }
            }
        }
    }
    vector<BakedPrim> bakedPrims;
    bakedPrims.reserve(numPrims);

    // Count the total number of vertices and start filling in the BakedPrim structs.
    int numVertices = 0, numIndices = 0;
    for (cgltf_size i = 0; i < sourceAsset->nodes_count; ++i) {
        const cgltf_node& node = sourceAsset->nodes[i];
        if (node.mesh) {
            for (cgltf_size j = 0; j < node.mesh->primitives_count; ++j) {
                const cgltf_primitive& sourcePrim = node.mesh->primitives[j];
                if (filterPrim(sourcePrim)) {
                    numVertices += sourcePrim.attributes[0].data->count;
                    numIndices += sourcePrim.indices->count;
                    bakedPrims.push_back({
                        .sourceNode = &node,
                        .sourceMesh = node.mesh,
                        .sourcePrimitive = &sourcePrim,
                    });
                }
            }
        }
    }

    // Allocate a buffer large enough to hold vertex positions and indices.
    const size_t vertexDataSize = sizeof(float3) * numVertices;
    const size_t indexDataSize = sizeof(uint32_t) * numIndices;
    uint8_t* bufferData = mStorage.bufferData.alloc(vertexDataSize + indexDataSize);
    float3* vertexData = (float3*) bufferData;
    uint32_t* indexData = (uint32_t*) (bufferData + vertexDataSize);

    // Next, perform the actual baking: convert all vertex positions to fp32, transform them by
    // their respective node matrix, etc.
    const cgltf_node* node = nullptr;
    mat4f matrix;
    for (size_t i = 0; i < numPrims; ++i) {
        BakedPrim& bakedPrim = bakedPrims[i];
        if (bakedPrim.sourceNode != node) {
            node = bakedPrim.sourceNode;
            cgltf_node_transform_world(node, &matrix[0][0]);
        }
        const cgltf_primitive* sourcePrim = bakedPrim.sourcePrimitive;
        bakedPrim.bakedPositions = vertexData;
        bakedPrim.bakedIndices = indexData;
        vertexData += sourcePrim->attributes[0].data->count;
        indexData += sourcePrim->indices->count;
        bakeTransform(&bakedPrim, matrix);
    }

    // We'll keep all the buffer views and accessors from the source asset and add a new pair of
    // views and accessors for each primitive: one for converted position data and one for converted
    // index data.
    size_t numBufferViews = sourceAsset->buffer_views_count + 2 * numPrims;
    size_t numAccessors = sourceAsset->accessors_count + 2 * numPrims;

    // Allocate memory for the various cgltf structures.
    cgltf_data* resultAsset = mStorage.resultAssets.alloc(1);
    cgltf_scene* scene = mStorage.scenes.alloc(1);
    cgltf_node** nodePointers = mStorage.nodePointers.alloc(numPrims);
    cgltf_node* nodes = mStorage.nodes.alloc(numPrims);
    cgltf_mesh* meshes = mStorage.meshes.alloc(numPrims);
    cgltf_primitive* prims = mStorage.prims.alloc(numPrims);
    cgltf_buffer_view* views = mStorage.views.alloc(numBufferViews);
    cgltf_accessor* accessors = mStorage.accessors.alloc(numAccessors);
    cgltf_attribute* attributes = mStorage.attributes.alloc(numAttributes);
    cgltf_buffer* buffers = mStorage.buffers.alloc(2);
    cgltf_image* images = mStorage.images.alloc(sourceAsset->images_count);

    // Populate the fields of the cgltf structures.
    cgltf_size positionsOffset = 0;
    cgltf_size indicesOffset = vertexDataSize;
    for (size_t primIndex = 0, attrIndex = 0; primIndex < numPrims; ++primIndex) {
        BakedPrim& bakedPrim = bakedPrims[primIndex];

        nodePointers[primIndex] = nodes + primIndex;

        nodes[primIndex] = {
            .name = bakedPrim.sourceNode->name,
            .mesh = meshes + primIndex,
        };

        meshes[primIndex] = {
            .name = bakedPrim.sourceMesh->name,
            .primitives = prims + primIndex,
            .primitives_count = 1,
        };

        cgltf_accessor& indicesAccessor = accessors[2 * primIndex] = {
            .component_type = cgltf_component_type_r_32u,
            .type = cgltf_type_scalar,
            .count = bakedPrim.sourcePrimitive->indices->count,
            .buffer_view = views + primIndex * 2,
        };

        cgltf_buffer_view& indicesBufferView = views[2 * primIndex] = {
            .buffer = buffers,
            .offset = indicesOffset,
            .size = indicesAccessor.count * sizeof(uint32_t)
        };
        indicesOffset += indicesBufferView.size;

        cgltf_accessor& positionsAccessor = accessors[2 * primIndex + 1] = {
            .component_type = cgltf_component_type_r_32f,
            .type = cgltf_type_vec3,
            .count = bakedPrim.sourcePrimitive->attributes[0].data->count,
            .buffer_view = views + primIndex * 2 + 1,
            .has_min = true,
            .has_max = true,
        };

        cgltf_buffer_view& positionsBufferView = views[2 * primIndex + 1] = {
            .buffer = buffers,
            .offset = positionsOffset,
            .size = positionsAccessor.count * sizeof(float3)
        };
        positionsOffset += positionsBufferView.size;

        *((float3*) positionsAccessor.min) = bakedPrim.bakedMin;
        *((float3*) positionsAccessor.max) = bakedPrim.bakedMax;

        cgltf_attribute& positionsAttribute = attributes[attrIndex++] = {
            .name = (char*) POSITION,
            .type = cgltf_attribute_type_position,
            .data = &positionsAccessor
        };
        size_t attrCount = bakedPrim.sourcePrimitive->attributes_count;
        for (size_t j = 1; j < attrCount; ++j) {
            auto& attr = attributes[attrIndex++] = bakedPrim.sourcePrimitive->attributes[j];
            size_t accessorIndex = attr.data - sourceAsset->accessors;
            attr.data = accessors + numPrims * 2 + accessorIndex;
        }

        prims[primIndex] = {
            .type = cgltf_primitive_type_triangles,
            .indices = &indicesAccessor,
            .material = bakedPrim.sourcePrimitive->material,
            .attributes = &positionsAttribute,
            .attributes_count = attrCount,
        };
    }

    scene->name = sourceAsset->scene->name;
    scene->nodes = nodePointers;
    scene->nodes_count = numPrims;

    buffers[0].size = vertexDataSize + indexDataSize;
    buffers[0].data = bufferData;

    buffers[1] = sourceAsset->buffers[0];

    resultAsset->file_type = sourceAsset->file_type;
    resultAsset->file_data = sourceAsset->file_data;
    resultAsset->asset = sourceAsset->asset;
    resultAsset->asset.generator = (char*) GENERATOR_ID;
    resultAsset->meshes = meshes;
    resultAsset->meshes_count = numPrims;
    resultAsset->accessors = accessors;
    resultAsset->accessors_count = numAccessors;
    resultAsset->buffer_views = views;
    resultAsset->buffer_views_count = numBufferViews;
    resultAsset->buffers = buffers;
    resultAsset->buffers_count = 2;
    resultAsset->nodes = nodes;
    resultAsset->nodes_count = numPrims;
    resultAsset->scenes = scene;
    resultAsset->scenes_count = 1;
    resultAsset->scene = scene;
    resultAsset->images = images;
    resultAsset->images_count = sourceAsset->images_count;
    resultAsset->textures = sourceAsset->textures;
    resultAsset->textures_count = sourceAsset->textures_count;
    resultAsset->materials = sourceAsset->materials;
    resultAsset->materials_count = sourceAsset->materials_count;
    resultAsset->samplers = sourceAsset->samplers;
    resultAsset->samplers_count = sourceAsset->samplers_count;

    // Copy over the buffer views, accessors, and textures, then fix up the pointers.
    const size_t offset = numPrims * 2;
    for (size_t i = offset; i < numBufferViews; ++i) {
        auto& view = views[i] = sourceAsset->buffer_views[i - offset];
        view.buffer = &buffers[1];
    }
    for (size_t i = offset; i < numAccessors; ++i) {
        auto& accessor = accessors[i] = sourceAsset->accessors[i - offset];
        size_t viewIndex = accessor.buffer_view - sourceAsset->buffer_views;
        accessor.buffer_view = views + offset + viewIndex;
    }

    for (size_t i = 0; i < sourceAsset->images_count; ++i) {
        auto& image = images[i] = sourceAsset->images[i];
        if (image.buffer_view) {
            size_t viewIndex = image.buffer_view - sourceAsset->buffer_views;
            image.buffer_view = views + offset + viewIndex;
        }
    }
    for (size_t i = 0; i < resultAsset->textures_count; ++i) {
        size_t imageIndex = resultAsset->textures[i].image - sourceAsset->images;
        resultAsset->textures[i].image = images + imageIndex;
    }

    return resultAsset;
}

void Pipeline::bakeTransform(BakedPrim* prim, const mat4f& transform) {
    const cgltf_primitive* source = prim->sourcePrimitive;
    const cgltf_attribute* sourcePositions = source->attributes;
    const size_t numVerts = sourcePositions->data->count;

    // Read position data, converting to float if necessary.
    cgltf_float* writePtr = &prim->bakedPositions->x;
    for (cgltf_size index = 0; index < numVerts; ++index, writePtr += 3) {
        cgltf_accessor_read_float(sourcePositions->data, index, writePtr, 3);
    }

    // Prepare for computing the post-transformed bounding box.
    float3& minpt = prim->bakedMin = std::numeric_limits<float>::max();
    float3& maxpt = prim->bakedMax = std::numeric_limits<float>::lowest();

    // Transform the positions and compute the new bounding box.
    float3* bakedPositions = prim->bakedPositions;
    for (cgltf_size index = 0; index < numVerts; ++index) {
        float3& pt = bakedPositions[index];
        pt = (transform * float4(pt, 1.0f)).xyz;
        minpt = min(minpt, pt);
        maxpt = max(maxpt, pt);
    }

    // Read index data, converting to uint32 if necessary.
    uint32_t* bakedIndices = prim->bakedIndices;
    for (cgltf_size index = 0, len = source->indices->count; index < len; ++index) {
        bakedIndices[index] = cgltf_accessor_read_index(source->indices, index);
    }
}

const cgltf_data* Pipeline::parameterize(const cgltf_data* sourceAsset) {
    auto atlas = xatlas::Create();

    utils::slog.e << "parameterize is not yet implemented." << utils::io::endl;

    xatlas::Destroy(atlas);
    atlas = nullptr;
    return nullptr;
}

void Pipeline::addSourceAsset(cgltf_data* asset) {
    mSourceAssets.push_back(asset);
}

Pipeline::~Pipeline() {
    for (auto asset : mSourceAssets) {
        cgltf_free(asset);
    }
}

} // anonymous namespace

namespace gltfio {

MeshPipeline::MeshPipeline() {
    mImpl = new Pipeline();
}

MeshPipeline::~MeshPipeline() {
    Pipeline* impl = (Pipeline*) mImpl;
    delete impl;
}

AssetHandle MeshPipeline::flatten(AssetHandle source, uint32_t flags) {
    Pipeline* impl = (Pipeline*) mImpl;
    const cgltf_data* asset = (const cgltf_data*) source;
    if (asset->buffers_count > 1) {
        asset = impl->flattenBuffers(asset);
    }
    asset = impl->flattenPrims(asset, flags);
    asset = impl->flattenBuffers(asset);
    return asset;
}

AssetHandle MeshPipeline::load(const utils::Path& fileOrDirectory) {
    utils::Path filename = fileOrDirectory;
    if (!filename.exists()) {
        utils::slog.e << "file " << filename << " not found!" << utils::io::endl;
        return nullptr;
    }
    if (filename.isDirectory()) {
        auto files = filename.listContents();
        for (auto file : files) {
            if (file.getExtension() == "gltf") {
                filename = file;
                break;
            }
        }
        if (filename.isDirectory()) {
            utils::slog.e << "no glTF file found in " << filename << utils::io::endl;
            return nullptr;
        }
    }

    // Peek at the file size to allow pre-allocation.
    long contentSize = static_cast<long>(getFileSize(filename.c_str()));
    if (contentSize <= 0) {
        utils::slog.e << "Unable to open " << filename << utils::io::endl;
        exit(1);
    }

    // Consume the glTF file.
    std::ifstream in(filename.c_str(), std::ifstream::in);
    vector<uint8_t> buffer(static_cast<unsigned long>(contentSize));
    if (!in.read((char*) buffer.data(), contentSize)) {
        utils::slog.e << "Unable to read " << filename << utils::io::endl;
        exit(1);
    }

    // Parse the glTF file.
    cgltf_options options { cgltf_file_type_gltf };
    cgltf_data* sourceAsset;
    cgltf_result result = cgltf_parse(&options, buffer.data(), contentSize, &sourceAsset);
    if (result != cgltf_result_success) {
        return nullptr;
    }
    Pipeline* impl = (Pipeline*) mImpl;
    impl->addSourceAsset(sourceAsset);

    // Load external resources.
    cgltf_load_buffers(&options, sourceAsset, filename.c_str());

    return sourceAsset;
}

void MeshPipeline::save(AssetHandle handle, const utils::Path& jsonPath,
        const utils::Path& binPath) {
    cgltf_data* asset = (cgltf_data*) handle;

    if (!isFlattened(asset)) {
        utils::slog.e << "Only flattened assets can be exported to disk." << utils::io::endl;
        return;
    }

    std::string binName = binPath.getName();
    asset->buffers[0].uri = (char*) (binName.c_str());
    cgltf_options options { cgltf_file_type_gltf };
    cgltf_write_file(&options, jsonPath.c_str(), asset);
    asset->buffers[0].uri = nullptr;

    std::ofstream binFile(binPath.c_str(), std::ios::binary);
    binFile.write((char*) asset->buffers[0].data, asset->buffers[0].size);
}

AssetHandle MeshPipeline::parameterize(AssetHandle source) {
    Pipeline* impl = (Pipeline*) mImpl;
    return impl->parameterize((const cgltf_data*) source);
}

}  // namespace gltfio
