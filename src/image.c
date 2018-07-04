#include "image.h"
#include "require.h"

#include <string.h>
#include <stdio.h>
#include <png.h>
#include <jpeglib.h>

void image_create(image_t* image,
                  size_t width,
                  size_t height,
                  size_t channels,
                  image_type_t type) {

    memset(image, 0, sizeof(image_t));

    image->width = width;
    image->height = height;
    image->channels = channels;
    
    image->num_elements = width*height*channels;

    image->type = type;

    image->bound_texture = 0;

    switch (type)  {
    case IMAGE_8U:
        image->element_size = sizeof(uint8_t);
        break;
    case IMAGE_32F:
        image->element_size = sizeof(float);
        break;
    case IMAGE_32U:
        image->element_size = sizeof(uint32_t);
        break;
    default:
        require( 0 && "bad image type in image_create!" );
    }

    size_t total_size = image->num_elements * image->element_size;

    buf_resize(&image->buf, total_size);

    image->data = image->buf.data;
               
}

//////////////////////////////////////////////////////////////////////

void image_destroy(image_t* image) {
    
    buf_free(&(image->buf));
    memset(image, 0, sizeof(image_t));
    
}

//////////////////////////////////////////////////////////////////////

void image_copy(image_t* dst, const image_t* src) {

    image_create(dst, src->width, src->height, src->channels, src->type);
    dst->bound_texture = src->bound_texture;

    require(dst->buf.size == src->buf.size);

    memcpy(dst->buf.data, src->buf.data, src->buf.size);

}

//////////////////////////////////////////////////////////////////////

void image8u_to_32f(const image_t* src,
                    image_t* dst) {

    require(src->type == IMAGE_8U);

    image_create(dst, src->width, src->height, src->channels, IMAGE_32F);

    for (size_t i=0; i<src->num_elements; ++i) {
        dst->data_32f[i] = (float)src->data_8u[i] / 255.0f;
    }

}

//////////////////////////////////////////////////////////////////////

void image32f_to_8u(const image_t* src,
                    image_t* dst) {

    require(src->type == IMAGE_32F);

    image_create(dst, src->width, src->height, src->channels, IMAGE_8U);
    
    for (size_t i=0; i<src->num_elements; ++i) {
        float f = src->data_32f[i];
        f = f < 0.0f ? 0.0f : f;
        f = f > 1.0f ? 1.0f : f;
        dst->data_8u[i] = (unsigned char)(f * 255.0f);
    }
    
}

//////////////////////////////////////////////////////////////////////

int write_png(const char* filename,
              const unsigned char* data, 
              size_t ncols,
              size_t nrows,
              size_t rowsz,
              int yflip,
              const float* pixel_scale) {

    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "error opening %s for output\n", filename);
        return 0;
    }
  
    png_structp png_ptr = png_create_write_struct
        (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr) {
        fprintf(stderr, "error creating write struct\n");
        fclose(fp);
        return 0;
    }
  
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fprintf(stderr, "error creating info struct\n");
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
        fclose(fp);
        return 0;
    }  

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "error processing PNG\n");
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return 0;
    }

    png_init_io(png_ptr, fp);

    png_set_IHDR(png_ptr, info_ptr, 
                 ncols, nrows,
                 8, 
                 PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);

    const float base_res = 72./.0254;

    if (pixel_scale) {
        
        int res_x = base_res * pixel_scale[0];
        int res_y = base_res * pixel_scale[1];

        png_set_pHYs(png_ptr, info_ptr,
                     res_x, res_y,
                     PNG_RESOLUTION_METER);

    }

    png_write_info(png_ptr, info_ptr);

    const unsigned char* rowptr = data + (yflip ? rowsz*(nrows-1) : 0);
    int rowdelta = rowsz * (yflip ? -1 : 1);

    for (size_t y=0; y<nrows; ++y) {
        png_write_row(png_ptr, (png_bytep)rowptr);
        rowptr += rowdelta;
    }

    png_write_end(png_ptr, info_ptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);

    /*
      fprintf(stderr, "wrote %s with size %dx%d\n", filename,
      (int)ncols, (int)nrows);
    */

    return 1;

}

//////////////////////////////////////////////////////////////////////

unsigned char* get_rowptr_and_delta(unsigned char* dstart,
                                    int height, int stride,
                                    int vflip,
                                    int* row_delta) {

    if (vflip) {
        *row_delta = -stride;
        return dstart + (height-1)*stride;
    } else {
        *row_delta = stride;
        return dstart;
    }

}

//////////////////////////////////////////////////////////////////////

void read_jpg(const buffer_t* raw,
              int vflip,
              image_t* dst_image) {

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr mgr;
    jmp_buf setjmp_buffer;

    cinfo.err = jpeg_std_error(&mgr);

    if (setjmp(setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
    }

    jpeg_create_decompress(&cinfo);

    jpeg_mem_src(&cinfo, (unsigned char*)raw->data, raw->size);

    int rc = jpeg_read_header(&cinfo, TRUE);

    if (rc != 1) {
        fprintf(stderr, "failure reading jpeg header!\n");
        exit(1);
    }

    jpeg_start_decompress(&cinfo);

    int width = cinfo.output_width;
    int height = cinfo.output_height;
    int pixel_size = cinfo.output_components;

    printf("jpeg is %dx%dx%d\n", width, height, pixel_size);

    if (width <= 0 || height <= 0 || (pixel_size != 3 && pixel_size != 1)) {
        fprintf(stderr, "incorrect JPG type!\n");
        exit(1);
    }

    int row_stride = width * 3;

    if (row_stride % 4) {
        fprintf(stderr, "warning: bad stride for GL_UNPACK_ALIGNMENT!\n");
    }
        
    image_create(dst_image, width, height, 3, IMAGE_8U);

    int row_delta;
    unsigned char* rowptr = get_rowptr_and_delta(dst_image->data_8u,
                                                 height, row_stride,
                                                 vflip, &row_delta);

    unsigned char* dummy = 0;
    if (pixel_size == 1) { dummy = malloc(row_stride); }

    while (cinfo.output_scanline < cinfo.output_height) {

        if (pixel_size == 1) {
            jpeg_read_scanlines(&cinfo, &dummy, 1);
            for (int x=0; x<width; ++x) {
                rowptr[3*x + 0] = rowptr[3*x + 1] = rowptr[3*x + 2] = dummy[x];
            }
        } else {
            jpeg_read_scanlines(&cinfo, &rowptr, 1);
        }
        
        rowptr += row_delta;
        
    }

    if (dummy) { free(dummy); }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

}

//////////////////////////////////////////////////////////////////////

typedef struct png_simple_stream {

    const unsigned char* start;
    size_t pos;
    size_t len;
    
} png_simple_stream_t;

//////////////////////////////////////////////////////////////////////

void png_stream_read(png_structp png_ptr,
                     png_bytep data,
                     png_size_t length) {


    png_simple_stream_t* str = (png_simple_stream_t*)png_get_io_ptr(png_ptr);

    require( str->pos + length <= str->len );

    memcpy(data, str->start + str->pos, length );
    str->pos += length;

}

//////////////////////////////////////////////////////////////////////

void read_png(const buffer_t* raw,
              int vflip,
              image_t* dst_image) {
    
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                                 NULL, NULL, NULL);

    if (!png_ptr) {
        fprintf(stderr, "error initializing png read struct!\n");
        exit(1);
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);

    if (!info_ptr) {
        fprintf(stderr, "error initializing png info struct!\n");
        exit(1);
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "PNG read error!\n");
        exit(1);
    }

    png_simple_stream_t str = { (const unsigned char*)raw->data, 0, raw->size };

    png_set_read_fn(png_ptr, &str, png_stream_read);

    png_set_sig_bytes(png_ptr, 0);
    png_read_info(png_ptr, info_ptr);

    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);
    int bitdepth = png_get_bit_depth(png_ptr, info_ptr);
    int channels = png_get_channels(png_ptr, info_ptr);
    int color_type = png_get_color_type(png_ptr, info_ptr);

    const char* color_type_str = "[unknown]";

#define HANDLE(x) case x: color_type_str = #x; break
    switch (color_type) {
        HANDLE(PNG_COLOR_TYPE_GRAY);
        HANDLE(PNG_COLOR_TYPE_GRAY_ALPHA);
        HANDLE(PNG_COLOR_TYPE_RGB);
        HANDLE(PNG_COLOR_TYPE_RGB_ALPHA);
        HANDLE(PNG_COLOR_TYPE_PALETTE);
    }

    printf("PNG is %dx%dx%d, color type %s\n",
           width, height, channels, color_type_str);

    if (bitdepth == 16) {
        png_set_strip_16(png_ptr);
        bitdepth = 8;
    }
    
    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        channels = 3;
        png_set_palette_to_rgb(png_ptr);
        if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
            channels = 4;
            png_set_tRNS_to_alpha(png_ptr);
        }
    }
    
    if (width <= 0 || height <= 0 || bitdepth != 8 ||
        (channels == 0 || channels > 4)) {
        
        fprintf(stderr, "invalid PNG settings!\n");
        exit(1);
        
    }

    int row_stride = width * channels;
    
    if (row_stride % 4) {
        fprintf(stderr, "warning: PNG data is not aligned for OpenGL!\n");
        exit(1);
    }


    image_create(dst_image, width, height, channels, IMAGE_8U);

    int row_delta;
    unsigned char* rowptr = get_rowptr_and_delta(dst_image->data_8u,
                                                 height, row_stride,
                                                 vflip, &row_delta);

    png_bytepp row_ptrs = malloc(height * sizeof(png_bytep));
    
    for (size_t i=0; i<height; ++i) {
        row_ptrs[i] = rowptr;
        rowptr += row_delta;
    }

    png_read_image(png_ptr, row_ptrs);

    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

    free(row_ptrs);

}

//////////////////////////////////////////////////////////////////////

const char* get_extension(const char* filename) {

    const char* extension = strrchr(filename, '.');
    if (!extension) { return ""; }

    return extension+1;

}

//////////////////////////////////////////////////////////////////////

void load_image(image_t* dst, const char* filename, int vflip) {

    const char* ext = get_extension(filename);

    int is_png = !strcasecmp(ext, "png");
    int is_jpg = (!strcasecmp(ext, "jpg") || !strcasecmp(ext, "jpeg"));

    if (!is_png && !is_jpg) {
        fprintf(stderr, "%s: unrecognized image format.\n", filename);
        exit(1);
    }

    buffer_t tmp = { 0, 0, 0 };
    buf_append_file(&tmp, filename, 1024*1024*64, 0);

    if (is_png) {
        read_png(&tmp, vflip, dst);
    } else {
        read_jpg(&tmp, vflip, dst);
    }

    buf_free(&tmp);
    
}

//////////////////////////////////////////////////////////////////////

void load_image_32f(image_t* dst, const char* filename, int vflip) {

    image_t tmp;
    load_image(&tmp, filename, vflip);

    image8u_to_32f(&tmp, dst);
    printf("converted %s to float\n", filename);
    
    image_destroy(&tmp);
    
    
}
