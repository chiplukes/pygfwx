/**
 * GFWX C Wrapper for Python ctypes
 *
 * This file provides C-compatible functions that wrap the GFWX C++ template API,
 * allowing the SDK to be called from Python via ctypes.
 */

#include "../gfwx.h"
#include <cstring>
#include <cstdint>

#ifdef _WIN32
    #define GFWX_API __declspec(dllexport)
#else
    #define GFWX_API __attribute__((visibility("default")))
#endif

extern "C" {

/**
 * Header structure matching Python ctypes definition.
 * All fields are int32_t for simplicity.
 */
struct GFWXHeaderC {
    int32_t sizex;
    int32_t sizey;
    int32_t layers;
    int32_t channels;
    int32_t bitDepth;
    int32_t quality;
    int32_t chromaScale;
    int32_t blockSize;
    int32_t filter;
    int32_t quantization;
    int32_t encoder;
    int32_t intent;
    int32_t version;
    int32_t isSigned;
};

/**
 * Convert C header to GFWX::Header
 */
static inline GFWX::Header headerFromC(const GFWXHeaderC* h) {
    return GFWX::Header(
        h->sizex, h->sizey, h->layers, h->channels, h->bitDepth,
        h->quality, h->chromaScale, h->blockSize, h->filter,
        h->quantization, h->encoder, h->intent
    );
}

/**
 * Convert GFWX::Header to C header
 */
static void headerToC(const GFWX::Header& src, GFWXHeaderC* dst) {
    dst->sizex = src.sizex;
    dst->sizey = src.sizey;
    dst->layers = src.layers;
    dst->channels = src.channels;
    dst->bitDepth = src.bitDepth;
    dst->quality = src.quality;
    dst->chromaScale = src.chromaScale;
    dst->blockSize = src.blockSize;
    dst->filter = src.filter;
    dst->quantization = src.quantization;
    dst->encoder = src.encoder;
    dst->intent = src.intent;
    dst->version = src.version;
    dst->isSigned = src.isSigned;
}

/**
 * Compress 8-bit unsigned image data.
 *
 * @param imageData     Pointer to image data (row-major, interleaved channels)
 * @param header        Pointer to header with compression parameters
 * @param buffer        Output buffer for compressed data
 * @param bufferSize    Size of output buffer in bytes
 * @param transform     Color transform program (nullptr for none, use GFWX_TRANSFORM_UYV etc)
 * @return              Size of compressed data in bytes, or negative error code
 */
GFWX_API int64_t gfwx_compress_u8(
    const uint8_t* imageData,
    GFWXHeaderC* header,
    uint8_t* buffer,
    size_t bufferSize,
    const int32_t* transform
) {
    GFWX::Header h = headerFromC(header);
    ptrdiff_t result = GFWX::compress(imageData, h, buffer, bufferSize, transform, nullptr, 0);
    headerToC(h, header);  // Update header with computed values (version, isSigned, etc.)
    return static_cast<int64_t>(result);
}

/**
 * Compress 16-bit unsigned image data.
 */
GFWX_API int64_t gfwx_compress_u16(
    const uint16_t* imageData,
    GFWXHeaderC* header,
    uint8_t* buffer,
    size_t bufferSize,
    const int32_t* transform
) {
    GFWX::Header h = headerFromC(header);
    ptrdiff_t result = GFWX::compress(imageData, h, buffer, bufferSize, transform, nullptr, 0);
    headerToC(h, header);
    return static_cast<int64_t>(result);
}

/**
 * Compress 8-bit signed image data.
 */
GFWX_API int64_t gfwx_compress_i8(
    const int8_t* imageData,
    GFWXHeaderC* header,
    uint8_t* buffer,
    size_t bufferSize,
    const int32_t* transform
) {
    GFWX::Header h = headerFromC(header);
    ptrdiff_t result = GFWX::compress(imageData, h, buffer, bufferSize, transform, nullptr, 0);
    headerToC(h, header);
    return static_cast<int64_t>(result);
}

/**
 * Compress 16-bit signed image data.
 */
GFWX_API int64_t gfwx_compress_i16(
    const int16_t* imageData,
    GFWXHeaderC* header,
    uint8_t* buffer,
    size_t bufferSize,
    const int32_t* transform
) {
    GFWX::Header h = headerFromC(header);
    ptrdiff_t result = GFWX::compress(imageData, h, buffer, bufferSize, transform, nullptr, 0);
    headerToC(h, header);
    return static_cast<int64_t>(result);
}

/**
 * Decompress to 8-bit unsigned image data.
 *
 * @param imageData     Output buffer for decompressed image (can be nullptr to read header only)
 * @param header        Pointer to header (will be filled with file header)
 * @param data          Compressed data
 * @param dataSize      Size of compressed data in bytes
 * @param downsampling  Downsampling level (0=full, 1=half, 2=quarter, etc.)
 * @param test          If true, just test if data is valid without full decode
 * @return              0 on success, positive = next point of interest, negative = error code
 */
GFWX_API int64_t gfwx_decompress_u8(
    uint8_t* imageData,
    GFWXHeaderC* header,
    const uint8_t* data,
    size_t dataSize,
    int32_t downsampling,
    int32_t test
) {
    GFWX::Header h;
    ptrdiff_t result = GFWX::decompress(imageData, h, data, dataSize, downsampling, test != 0);
    headerToC(h, header);
    return static_cast<int64_t>(result);
}

/**
 * Decompress to 16-bit unsigned image data.
 */
GFWX_API int64_t gfwx_decompress_u16(
    uint16_t* imageData,
    GFWXHeaderC* header,
    const uint8_t* data,
    size_t dataSize,
    int32_t downsampling,
    int32_t test
) {
    GFWX::Header h;
    ptrdiff_t result = GFWX::decompress(imageData, h, data, dataSize, downsampling, test != 0);
    headerToC(h, header);
    return static_cast<int64_t>(result);
}

/**
 * Decompress to 8-bit signed image data.
 */
GFWX_API int64_t gfwx_decompress_i8(
    int8_t* imageData,
    GFWXHeaderC* header,
    const uint8_t* data,
    size_t dataSize,
    int32_t downsampling,
    int32_t test
) {
    GFWX::Header h;
    ptrdiff_t result = GFWX::decompress(imageData, h, data, dataSize, downsampling, test != 0);
    headerToC(h, header);
    return static_cast<int64_t>(result);
}

/**
 * Decompress to 16-bit signed image data.
 */
GFWX_API int64_t gfwx_decompress_i16(
    int16_t* imageData,
    GFWXHeaderC* header,
    const uint8_t* data,
    size_t dataSize,
    int32_t downsampling,
    int32_t test
) {
    GFWX::Header h;
    ptrdiff_t result = GFWX::decompress(imageData, h, data, dataSize, downsampling, test != 0);
    headerToC(h, header);
    return static_cast<int64_t>(result);
}

/**
 * Read header only (no decompression).
 * Pass nullptr for imageData to just read the header.
 */
GFWX_API int64_t gfwx_read_header(
    GFWXHeaderC* header,
    const uint8_t* data,
    size_t dataSize
) {
    GFWX::Header h;
    // Pass nullptr as imageData to just read header
    ptrdiff_t result = GFWX::decompress(static_cast<uint8_t*>(nullptr), h, data, dataSize, 0, false);
    headerToC(h, header);
    return static_cast<int64_t>(result);
}

/**
 * Calculate required buffer size for image data.
 */
GFWX_API size_t gfwx_buffer_size(const GFWXHeaderC* header) {
    GFWX::Header h = headerFromC(header);
    h.isSigned = header->isSigned;
    return h.bufferSize();
}

/**
 * Get UYV transform program (for RGB images).
 * Returns pointer to static array, do not free.
 */
GFWX_API const int32_t* gfwx_transform_uyv() {
    static const int32_t transform[] = GFWX_TRANSFORM_UYV;
    return transform;
}

/**
 * Get A710 transform program for RGB images.
 */
GFWX_API const int32_t* gfwx_transform_a710_rgb() {
    static const int32_t transform[] = GFWX_TRANSFORM_A710_RGB;
    return transform;
}

/**
 * Get A710 transform program for BGR images.
 */
GFWX_API const int32_t* gfwx_transform_a710_bgr() {
    static const int32_t transform[] = GFWX_TRANSFORM_A710_BGR;
    return transform;
}

// Constants for Python
GFWX_API int32_t gfwx_quality_max() { return GFWX::QualityMax; }
GFWX_API int32_t gfwx_filter_linear() { return GFWX::FilterLinear; }
GFWX_API int32_t gfwx_filter_cubic() { return GFWX::FilterCubic; }
GFWX_API int32_t gfwx_encoder_turbo() { return GFWX::EncoderTurbo; }
GFWX_API int32_t gfwx_encoder_fast() { return GFWX::EncoderFast; }
GFWX_API int32_t gfwx_encoder_contextual() { return GFWX::EncoderContextual; }
GFWX_API int32_t gfwx_result_ok() { return GFWX::ResultOk; }
GFWX_API int32_t gfwx_error_overflow() { return GFWX::ErrorOverflow; }
GFWX_API int32_t gfwx_error_malformed() { return GFWX::ErrorMalformed; }
GFWX_API int32_t gfwx_error_type_mismatch() { return GFWX::ErrorTypeMismatch; }
GFWX_API int32_t gfwx_error_unsupported() { return GFWX::ErrorUnsupported; }

// Intent constants
GFWX_API int32_t gfwx_intent_generic() { return GFWX::IntentGeneric; }
GFWX_API int32_t gfwx_intent_mono() { return GFWX::IntentMono; }
GFWX_API int32_t gfwx_intent_rgb() { return GFWX::IntentRGB; }
GFWX_API int32_t gfwx_intent_rgba() { return GFWX::IntentRGBA; }
GFWX_API int32_t gfwx_intent_bgr() { return GFWX::IntentBGR; }
GFWX_API int32_t gfwx_intent_bgra() { return GFWX::IntentBGRA; }

}  // extern "C"
