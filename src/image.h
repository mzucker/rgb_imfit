#ifndef _IMAGE_H_
#define _IMAGE_H_

#include "buffer.h"
#include <stdint.h>

typedef enum image_type {
    IMAGE_8U,
    IMAGE_32F,
    IMAGE_32U
} image_type_t;

typedef struct image {
    
    size_t width, height, channels, num_elements;

    image_type_t type;
    size_t element_size;

    uint32_t bound_texture;
    
    buffer_t buf;
    
    union {
        void*     data;
        uint8_t*  data_8u;
        float*    data_32f;
        uint32_t* data_32u;
    };
    
} image_t;

void image_create(image_t* image,
                  size_t width,
                  size_t height,
                  size_t channels,
                  image_type_t type);

void image_destroy(image_t* image);

void image_copy(image_t* dst, const image_t* src);

void image8u_to_32f(const image_t* src, image_t* dst);

void image32f_to_8u(const image_t* src, image_t* dst);

int write_png(const char* filename,
              const unsigned char* data, 
              size_t ncols,
              size_t nrows,
              size_t rowsz,
              int yflip,
              const float* pixel_scale);

void read_jpg(const buffer_t* raw,
              int vflip,
              image_t* dst_image);

void read_png(const buffer_t* raw,
              int vflip,
              image_t* dst_image);

const char* get_extension(const char* filename);

void load_image(image_t* dst, const char* filename, int vflip);

void load_image_32f(image_t* dst, const char* filename, int vflip);

#endif
