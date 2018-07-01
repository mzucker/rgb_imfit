#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <png.h>
#include <jpeglib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/*  TODO: 

 - ESC to quit?
 - GA instead of annealing?
 - figure out how to better capture fine detail?
 - resize input using mipmapping

 */

//////////////////////////////////////////////////////////////////////
// TYPEDEFS/ENUMS

typedef struct buffer {
    
    char*  data;
    size_t alloc;
    size_t size;
    
} buffer_t;

enum {
    BUF_RAW_APPEND = 0,
    BUF_NULL_TERMINATE = 1
};

typedef enum image_type {
    IMAGE_8U,
    IMAGE_32F
} image_type_t;

typedef struct image {
    
    size_t width, height, channels, num_elements;

    image_type_t type;
    size_t element_size;

    GLuint bound_texture;
    
    buffer_t buf;
    
    union {
        void* data;
        unsigned char* data_8u;
        float* data_32f;
    };
    
} image_t;


enum {
    VIS_TILES = 2,
    MAX_INPUTS = 4,
    MAX_UNIFORM_UPDATES = 4,
};

typedef void (*glUniformFloatFunc)(GLint, GLsizei, const GLfloat*);
typedef void (*glUniformIntFunc)(GLint, GLsizei, const GLint*);

typedef struct uniform_update {
    
    GLenum type; // GL_NONE, GL_INT, GL_FLOAT
    
    const char* name;
    int array_length;
    
    union {
        const float* src_float;
        const int*   src_int;
    };
    
    union {
        glUniformIntFunc   int_func;
        glUniformFloatFunc float_func;
    };
    
} uniform_update_t;

typedef struct framebuffer {

    const char* name;

    size_t width, height;
    GLenum internal_format;

    GLuint render_texture;
    GLuint framebuffer;

    size_t num_inputs;
    GLenum inputs[MAX_INPUTS];

    GLuint program;

    size_t num_uupdates;
    uniform_update_t uupdates[MAX_UNIFORM_UPDATES];

    int request_screenshot;

} framebuffer_t;

typedef struct { uint64_t state;  uint64_t inc; } pcg32_random_t;

#define PCG32_INITIALIZER   { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL }

pcg32_random_t rng_global = PCG32_INITIALIZER;

//////////////////////////////////////////////////////////////////////
// GLOBALS

enum {
    MAX_SOURCE_LENGTH = 1024*1024,
    SUM_WINDOW_SIZE = 2
};

typedef enum solver_type {
    SOLVER_ANNEALING,
    SOLVER_GA,
} solver_type_t;

buffer_t vertex_src = { 0, 0, 0 };

size_t gabors_per_tile = 128;

#ifdef __APPLE__
size_t num_tiles = 100;
size_t num_profile = 10;
#else
size_t num_tiles = 100;
size_t num_profile = 1000;
#endif

char common_defines[1024] = "";

solver_type_t solver;

GLuint vertex_shader, gabor_fragment_shader;

image_t src_image32f;
image_t weight_image32f;
image_t palette_image32f;

image_t param_image32f;

float px;

framebuffer_t gabor_eval_fb;
framebuffer_t gabor_sum_fb;
framebuffer_t gabor_compare_fb;

size_t num_reduce = 0;
framebuffer_t* reduce_fbs = 0;
char* reduce_names = 0;

framebuffer_t main_fb;

image_t reduced_image32f;
float* objective_values;

typedef struct anneal_info {
    float prev_cost;
    double t_max;
    double t_rate;
    double change_fraction;
    float p_reinitialize;
    float mutate_amount;
    image_t good_params32f;
} anneal_info_t;

anneal_info_t anneal;

enum {
    GABOR_PARAM_U = 0,
    GABOR_PARAM_V,
    GABOR_PARAM_S,
    GABOR_PARAM_T,
    GABOR_PARAM_PHI0,
    GABOR_PARAM_PHI1,
    GABOR_PARAM_PHI2,
    GABOR_PARAM_R,
    GABOR_PARAM_H0,
    GABOR_PARAM_H1,
    GABOR_PARAM_H2,
    GABOR_PARAM_L,
    GABOR_NUM_PARAMS,
    GABOR_NUM_NORMAL = 6
};


const int normal_params[GABOR_NUM_NORMAL] = {
    GABOR_PARAM_U,
    GABOR_PARAM_V,
    GABOR_PARAM_PHI0,
    GABOR_PARAM_PHI1,
    GABOR_PARAM_PHI2,
    GABOR_PARAM_R,
};


const char* param_names[GABOR_NUM_PARAMS] = {
    "u", "v", "s", "t",
    "phi0", "phi1", "phi2", "r",
    "h0", "h1", "h2", "l"
};

const float t0_scl = 1;
const float t1_scl = 4;

const float l0_scl = 2;
const float l1_scl = 8;

// u, v, s, t | phi[3], r | h[3], l
float param_bounds[GABOR_NUM_PARAMS][2] = {
    { -2, 2 },
    { -2, 2 },
    { 0, 2 }, // s special
    { 0, 1 }, // t special
    { 0, 2*M_PI },
    { 0, 2*M_PI },
    { 0, 2*M_PI },
    { 0, 2*M_PI },
    { 0, 2 }, // h0 special
    { 0, 2 }, // h1 special
    { 0, 2 }, // h2 special
    { 0, 1 } // l special
};

//////////////////////////////////////////////////////////////////////

#define require(x) do { if (!(x)) { _require_fail(__FILE__, __LINE__, #x); } } while (0)

void _require_fail(const char* file, int line, const char* what) {

    fprintf(stderr, "%s:%d: requirement failed: %s\n",
            file, line, what);

    exit(1);
    
}

//////////////////////////////////////////////////////////////////////

uint32_t pcg32_random_r(pcg32_random_t* rng) {
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

uint32_t pcg32_random() {
    return pcg32_random_r(&rng_global);
}

void pcg32_srandom_r(pcg32_random_t* rng,
                     uint64_t initstate,
                     uint64_t initseq) {

    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    pcg32_random_r(rng);
    rng->state += initstate;
    pcg32_random_r(rng);
    
}

//////////////////////////////////////////////////////////////////////

float random_float() {
    return (float)pcg32_random() / (float)UINT32_MAX;
}

void random_float_sign(float* x, float* s) {
    uint32_t r = pcg32_random();
    *s = (r & 1) ? -1 : 1;
    *x = (float)(r >> 1) / (float)(UINT32_MAX >> 1);
}

float lerp(float a, float b, float u) {
    return a + u*(b-a);
}

float signed_random() {
    return random_float()*2-1;
}

float signed_random2() {
    float x, s;
    random_float_sign(&x, &s);
    return x*x*s;
}

float signed_random3() {
    float x, s;
    random_float_sign(&x, &s);
    return x*x*x*s;
}

//////////////////////////////////////////////////////////////////////

float clamp(float x, float minval, float maxval) {
    return x < minval ? minval : x > maxval ? maxval : x;
}

float wrap2pi(float x) {
    return fmod(x, 2*M_PI);
}

#define MAX(a,b) ((a) > (b) ? (a) : (b))

//////////////////////////////////////////////////////////////////////

void buf_grow(buffer_t* buf, size_t len) {

    size_t new_size = buf->size + len;
    
    if (!buf->data) {
        
        buf->data = malloc(len);
        buf->alloc = len;

    } else if (buf->alloc < new_size) {

        while (buf->alloc < new_size) {
            buf->alloc *= 2;
        }

        buf->data = realloc(buf->data, buf->alloc);

    }

    require(buf->alloc >= new_size);

}

//////////////////////////////////////////////////////////////////////

void buf_resize(buffer_t* buf, size_t size) {

    if (buf->alloc < size) {
        buf_grow(buf, size - buf->alloc);
    }

    require(buf->alloc >= size);

    buf->size = size;

}

//////////////////////////////////////////////////////////////////////

void buf_free(buffer_t* buf) {

    if (buf->data) { free(buf->data); }
    memset(buf, 0, sizeof(buffer_t));

}

//////////////////////////////////////////////////////////////////////

void buf_append_mem(buffer_t* buf, const void* src,
                    size_t len, int null_terminate) {

    buf_grow(buf, len + (null_terminate ? 1 : 0));
    
    memcpy(buf->data + buf->size, src, len);
    
    buf->size += len;
    if (null_terminate) { buf->data[buf->size] = 0; }

}

//////////////////////////////////////////////////////////////////////

void buf_append_file(buffer_t* buf, const char* filename,
                     size_t max_length, int null_terminate) {

    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "error opening %s\n\n", filename);
        exit(1);
    }
    
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);

    if (fsize < 0 || fsize > max_length) {
        fprintf(stderr, "file exceeds maximum size!\n\n");
        exit(1);
    }
  
    fseek(fp, 0, SEEK_SET);
    
    buf_grow(buf, fsize + (null_terminate ? 1 : 0));

    int nread = fread(buf->data + buf->size, fsize, 1, fp);

    if (nread != 1) {
        fprintf(stderr, "error reading %s\n\n", filename);
        exit(1);
    }

    buf->size += fsize;
    if (null_terminate) { buf->data[buf->size] = 0; }

}

//////////////////////////////////////////////////////////////////////

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
        image->element_size = 1;
        break;
    case IMAGE_32F:
        image->element_size = sizeof(float);
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

//////////////////////////////////////////////////////////////////////

void dieusage() {

    fprintf(stderr,
            "usage: rgb_imfit INPUTIMAGE [WEIGHTIMAGE]\n");

    exit(1);

}

//////////////////////////////////////////////////////////////////////

void get_options(int argc, char** argv) {

    int cur_arg = 1;
    
    if (argc != cur_arg+1 && argc != cur_arg+2) {
        dieusage();
    }

    load_image_32f(&src_image32f, argv[cur_arg], GL_TRUE);

    if (cur_arg + 1 < argc) {
        
        load_image_32f(&weight_image32f, argv[cur_arg+1], GL_TRUE);

        printf("weight_image32f.width = %zu, height=%zu, channels=%zu\n",
               weight_image32f.width,
               weight_image32f.height,
               weight_image32f.channels);
        
        require(weight_image32f.width == src_image32f.width &&
                weight_image32f.height == src_image32f.height &&
                weight_image32f.channels == 1);
        
    } else {

        image_create(&weight_image32f,
                     src_image32f.width,
                     src_image32f.height,
                     1, IMAGE_32F);

        for (size_t i=0; i<weight_image32f.num_elements; ++i) {
            weight_image32f.data_32f[i] = 1;
        }
        
    }

    solver = SOLVER_ANNEALING;
    num_tiles = 1;


}

//////////////////////////////////////////////////////////////////////

const char* get_error_string(GLenum error) {
    switch (error) {
    case GL_NO_ERROR:
        return "no error";
    case GL_INVALID_ENUM:
        return "invalid enum";
    case GL_INVALID_VALUE:
        return "invalid value";
    case GL_INVALID_OPERATION:
        return "invalid operation";
    case GL_INVALID_FRAMEBUFFER_OPERATION:
        return "invalid framebuffer operation";
    case GL_OUT_OF_MEMORY:
        return "out of memory";
    default:
        return "unknown error";
    }
}

//////////////////////////////////////////////////////////////////////

void check_opengl_errors(const char* context) { 
    GLenum error = glGetError();
    if (!context || !*context) { context = "error"; }
    if (error) {
        fprintf(stderr, "%s: %s\n", context, get_error_string(error));
        exit(1);
                                                                         
    }
}

//////////////////////////////////////////////////////////////////////

void error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW error: %s\n", description);
}

//////////////////////////////////////////////////////////////////////

GLenum gl_datatype(image_type_t type) {
    require( type == IMAGE_8U || type == IMAGE_32F );
    if (type == IMAGE_8U) {
        return GL_UNSIGNED_BYTE;
    } else {
        return GL_FLOAT;
    }
}

//////////////////////////////////////////////////////////////////////

GLenum gl_format(size_t channels) {
    require( channels == 1 || channels == 3 || channels == 4 );
    if (channels == 1) {
        return GL_RED;
    } else if (channels == 3) {
        return GL_RGB;
    } else {
        return GL_RGBA;
    }
}

//////////////////////////////////////////////////////////////////////

GLenum gl_internal_format(GLenum format, GLenum datatype) {

    require(format == GL_RED || format == GL_RGB || format == GL_RGBA);
    require(datatype == GL_UNSIGNED_BYTE || datatype == GL_FLOAT);

    if (format == GL_RED) {
        return datatype == GL_FLOAT ? GL_R32F : GL_R8;
    } else if (format == GL_RGB) {
        return datatype == GL_FLOAT ? GL_RGB32F : GL_RGB8;
    } else {
        return datatype == GL_FLOAT ? GL_RGBA32F : GL_RGBA8;
    }
    
}

//////////////////////////////////////////////////////////////////////

void upload_texture(const image_t* image) {

    GLenum format = gl_format(image->channels);
    GLenum datatype = gl_datatype(image->type);

    glBindTexture(GL_TEXTURE_2D, image->bound_texture);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    image->width, image->height,
                    format, datatype,
                    image->buf.data);
    
    check_opengl_errors("upload texture32f!");


}

//////////////////////////////////////////////////////////////////////

GLuint make_texture(image_t* image) {

    GLuint texname;
    
    glGenTextures(1, &texname);
    glBindTexture(GL_TEXTURE_2D, texname);

    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MAG_FILTER,
                    GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_WRAP_S,
                    GL_CLAMP_TO_EDGE);

    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_WRAP_T,
                    GL_CLAMP_TO_EDGE);
    
    GLenum format = gl_format(image->channels);
    GLenum datatype = gl_datatype(image->type);
    GLenum internal_format = gl_internal_format(format, datatype);
        
    glTexStorage2D(GL_TEXTURE_2D, 1, internal_format,
                   image->width, image->height);

    check_opengl_errors("glTexStorage2D");
    
    image->bound_texture = texname;
    upload_texture(image);


    return texname;

}

//////////////////////////////////////////////////////////////////////

void read_pixels(image_t* image) {

    size_t w = image->width, h = image->height;

    GLenum format = gl_format(image->channels);
    GLenum datatype = gl_datatype(image->type);
    
    glReadPixels(0, 0, w, h, format, datatype, image->data);

    check_opengl_errors("read_pixels");

}

//////////////////////////////////////////////////////////////////////

GLFWwindow* setup_window() {

    if (!glfwInit()) {
        fprintf(stderr, "Error initializing GLFW!\n");
        exit(1);
    }

    glfwSetErrorCallback(error_callback);

#ifdef __APPLE__    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);    
#endif

    glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);

    int w = VIS_TILES*src_image32f.width, h = src_image32f.height;

    int k = 1600/w;
    
    if (k > 1) {
        w *= k;
        h *= k;
    }

    printf("creating window of size %d %d\n", w, h);

    GLFWwindow* window = glfwCreateWindow(w, h,
                                          "foo", NULL, NULL);

    glfwMakeContextCurrent(window);
#ifdef ST_GLFW_USE_GLEW
    glewInit();
#endif
    glfwSwapInterval(0);
    
    check_opengl_errors("after setting up glfw & glew");

    return window;

}

//////////////////////////////////////////////////////////////////////

GLuint make_shader(GLenum type,
                   GLint count,
                   const char** srcs) {

    GLint length[count];

    for (GLint i=0; i<count; ++i) {
        length[i] = strlen(srcs[i]);
    }

    GLuint shader = glCreateShader(type);
    glShaderSource(shader, count, srcs, length);
    glCompileShader(shader);
  
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

    if (!status) {
        char buf[4096];
        glGetShaderInfoLog(shader, sizeof(buf), NULL, buf);
        fprintf(stderr, "error compiling %s shader:\n\n%s\n",
                type == GL_VERTEX_SHADER ? "vertex" : "fragment",
                buf);
        exit(1);
    }

    return shader;
  
}

//////////////////////////////////////////////////////////////////////

GLuint make_shader_file(GLenum type,
                        const char* filename,
                        size_t max_size) {

    buffer_t src = { 0, 0, 0 };

    buf_append_file(&src, filename, 1024, BUF_NULL_TERMINATE);

    const char* srcstrs[1] = { src.data };

    GLuint shader = make_shader(GL_VERTEX_SHADER, 1, srcstrs);

    buf_free(&src);

    return shader;

}
    

//////////////////////////////////////////////////////////////////////
    
void fb_setup(framebuffer_t* fb,
              const char* name,
              size_t width, size_t height,
              GLenum internal_format,
              const char* fragment_filename,
              size_t max_length) {

    memset(fb, 0, sizeof(*fb));

    printf("setting up framebuffer %s with size %dx%d\n",
           name, (int)width, (int)height);

    fb->name = name;
    
    fb->width = width;
    fb->height = height;
    fb->internal_format = internal_format;

        
    fb->num_inputs = 0;
    fb->program = 0;

    if (fb->internal_format != GL_NONE) {
    
        glGenTextures(1, &fb->render_texture);
        glGenFramebuffers(1, &fb->framebuffer);

        glBindTexture(GL_TEXTURE_2D, fb->render_texture);

        glTexStorage2D(GL_TEXTURE_2D, 1, internal_format,
                       width, height);
            
        glBindFramebuffer(GL_FRAMEBUFFER, fb->framebuffer);
    
        glFramebufferTexture2D(GL_FRAMEBUFFER,
                               GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D,
                               fb->render_texture,
                               0);
    
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

        require(status == GL_FRAMEBUFFER_COMPLETE);

    } else {

        fb->render_texture = 0;
        fb->framebuffer = 0;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

    }

    check_opengl_errors("setup framebuffer");

    buffer_t src = { 0, 0, 0 };

    buf_append_file(&src, fragment_filename, max_length, BUF_NULL_TERMINATE);
    
    const char* fsrc[3] = {
        "#version 330\n",
        common_defines,
        src.data,
    };

    GLuint fragment_shader = make_shader(GL_FRAGMENT_SHADER, 3, fsrc);

    buf_free(&src);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    glUseProgram(program);

    /*
    GLint count;
    glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &count);
    printf("Active Uniforms: %d\n", count);

    for (GLint i = 0; i < count; i++) {
        char name[1024];
        GLsizei length;
        GLint size;
        GLenum type;
        glGetActiveUniform(program, i, 1024, &length, &size, &type, name);
        printf("Uniform #%d Type: %u Name: %s\n", i, type, name);
    }    
    */
    

    GLint vpos_location = glGetAttribLocation(program, "vertexPosition");
    require(vpos_location != -1);
    
    glEnableVertexAttribArray(vpos_location);
    glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE,
                          sizeof(float) * 2, (void*) 0);
    
    check_opengl_errors("make program");

    fb->program = program;
    
    
}

//////////////////////////////////////////////////////////////////////

void fb_add_input(framebuffer_t* fb,
                  const char* uniform_name,
                  GLuint texname) {

    require(fb->num_inputs < MAX_INPUTS);

    glUseProgram(fb->program);
    check_opengl_errors("before glGetUniformLocation");
    
    GLint param_location = glGetUniformLocation(fb->program, uniform_name);

    if (param_location != -1) {
        glUniform1i(param_location, fb->num_inputs);
        printf("  hooked up input %s for %s\n", uniform_name, fb->name);
    } else {
        fprintf(stderr, "warning: input %s is unused in program for %s\n",
                uniform_name, fb->name);
    }
    
    fb->inputs[fb->num_inputs++] = texname;
    
}

//////////////////////////////////////////////////////////////////////

void fb_screenshot(const framebuffer_t* fb) {
    
    size_t w = fb->width, h = fb->height;

    int stride = w*3;

    int align;
    glGetIntegerv(GL_PACK_ALIGNMENT, &align);

    if (stride % align) {
        stride += align - stride % align;
    }

    unsigned char* screen = (unsigned char*)malloc(h*stride);
  
    if (!screen) {
        fprintf(stderr, "out of memory allocating screen!\n");
        exit(1);
    }
  
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, screen);

    check_opengl_errors("after glReadPixels");

    char filename[1024];

    if (fb->request_screenshot <= 0) {
        snprintf(filename, 1024, "%s.png", fb->name);
    } else {
        snprintf(filename, 1024, "%s%06d.png", fb->name, fb->request_screenshot-1);
    }
  
    write_png(filename, screen, w, h, stride, GL_TRUE, NULL);

    free(screen);

  
}

//////////////////////////////////////////////////////////////////////

void fb_enqueue_uupdate_i(framebuffer_t* fb,
                          const char* name, 
                          int array_length,
                          const int* src_int,
                          glUniformIntFunc int_func) {

    require(fb->num_uupdates < MAX_UNIFORM_UPDATES);

    uniform_update_t* uu = fb->uupdates + fb->num_uupdates;

    ++fb->num_uupdates;

    uu->type = GL_INT;
    uu->name = name;
    uu->array_length = array_length;
    uu->src_int = src_int;
    uu->int_func = int_func;

}

//////////////////////////////////////////////////////////////////////

void fb_enqueue_uupdate_f(framebuffer_t* fb,
                          const char* name, 
                          int array_length,
                          const float* src_float,
                          glUniformFloatFunc float_func) {

    require(fb->num_uupdates < MAX_UNIFORM_UPDATES);

    uniform_update_t* uu = fb->uupdates + fb->num_uupdates;

    ++fb->num_uupdates;

    uu->type = GL_FLOAT;
    uu->name = name;
    uu->array_length = array_length;
    uu->src_float = src_float;
    uu->float_func = float_func;

}

//////////////////////////////////////////////////////////////////////

void fb_run_uupdates(framebuffer_t* fb) {
    
    for (size_t i=0; i<fb->num_uupdates; ++i) {

        const uniform_update_t* uu = fb->uupdates + i;
        require(uu->type == GL_INT || uu->type == GL_FLOAT);

        int location = glGetUniformLocation(fb->program, uu->name);
        if (location == -1) { continue; }
        
        if (uu->type == GL_INT) {
            uu->int_func(location, uu->array_length, uu->src_int);
        } else {
            uu->float_func(location, uu->array_length, uu->src_float);
        }
        
    }
    
    memset(fb->uupdates, 0, sizeof(fb->uupdates));
    fb->num_uupdates = 0;

}

//////////////////////////////////////////////////////////////////////

void fb_draw(framebuffer_t* fb) {

    glBindFramebuffer(GL_FRAMEBUFFER, fb->framebuffer);
    glViewport(0, 0, fb->width, fb->height);
    glUseProgram(fb->program);

    fb_run_uupdates(fb);
    
    check_opengl_errors("use program and do uniform updates");
    
    const GLfloat zero[4] = { 0, 0, 0, 0 };
    glClearBufferfv(GL_COLOR, 0, zero);
    
    for (size_t i=0; i<fb->num_inputs; ++i) {
        
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, fb->inputs[i]);

    }

    if (fb->internal_format == GL_NONE) {
        // main
        GLint odims[2] = { fb->width, fb->height };
        GLint output_dims = glGetUniformLocation(fb->program, "outputDims");
        glUniform2iv(output_dims, 1, odims);
    }
    

    check_opengl_errors("use framebuffer");

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, (void*)0);

    check_opengl_errors("draw framebuffer");

    if (fb->request_screenshot) {
        fb_screenshot(fb);
        fb->request_screenshot = 0;
    }
    
}

//////////////////////////////////////////////////////////////////////

void draw_main(GLFWwindow* window) {

    GLuint old_framebuffer = main_fb.framebuffer;
    
    size_t old_width = main_fb.width;
    size_t old_height = main_fb.height;

    if (window) {

        GLint w, h;
        glfwGetFramebufferSize(window, &w, &h);
        
        main_fb.framebuffer = 0;
        main_fb.width = w;
        main_fb.height = h;

    }

    GLint odims[2] = { main_fb.width, main_fb.height };

    fb_enqueue_uupdate_i(&main_fb, "outputDims", 1, odims,
                         glUniform2iv);
    
    fb_draw(&main_fb);

    if (window) {
        
        main_fb.framebuffer = old_framebuffer;
        main_fb.width = old_width;
        main_fb.height = old_height;

    }

    
}

//////////////////////////////////////////////////////////////////////

void init_params(float pi[GABOR_NUM_PARAMS]) {

    const float s0 = param_bounds[GABOR_PARAM_S][0];
    const float s1 = param_bounds[GABOR_PARAM_S][1];
    const float h1 = param_bounds[GABOR_PARAM_H0][1];
    
    for (int j=0; j<GABOR_NUM_PARAMS; ++j) {
        pi[j] = random_float();
    }

    for (int k=0; k<GABOR_NUM_NORMAL; ++k) {
        int j = normal_params[k];
        pi[j] = lerp(param_bounds[j][0], param_bounds[j][1], pi[j]);
    }

    for (int k=0; k<3; ++k) {
        float h = pi[GABOR_PARAM_H0+k];
        h = h * h * h;
        h *= h1 * 4. / sqrt(gabors_per_tile);
        pi[GABOR_PARAM_H0+k] = h;
    }

    float y = pi[GABOR_PARAM_S];
    float x = y*y*y; //cbrt( y*(s13 - s03) + s03 )
    float s = lerp(s0, s1, x);
    float t = s * lerp(t0_scl, t1_scl, pi[GABOR_PARAM_T]);
    float l = s * lerp(l0_scl, l1_scl, pi[GABOR_PARAM_L]);

    pi[GABOR_PARAM_S] = s;
    pi[GABOR_PARAM_T] = t;
    pi[GABOR_PARAM_L] = l;
        
    require(l >= l0_scl*s);
    require(l <= l1_scl*s);
        
    require(t >= t0_scl*s);
    require(t <= t1_scl*s);

}


//////////////////////////////////////////////////////////////////////

void mutate_params(float pi[GABOR_NUM_PARAMS], float amount) {

    float mybounds[GABOR_NUM_PARAMS][2];
    memcpy(mybounds, param_bounds, sizeof(mybounds));

    for (int j=0; j<GABOR_NUM_PARAMS; ++j) {
        
        const float* lohi = mybounds[j];
        pi[j] += amount * (lohi[1]-lohi[0]) * signed_random3();

        if (j < GABOR_PARAM_PHI0 || j > GABOR_PARAM_R) {
            pi[j] = clamp(pi[j], lohi[0], lohi[1]);
        } else {
            pi[j] = wrap2pi(pi[j]);
        }
        
        if (j == GABOR_PARAM_S) {
            float s = pi[j];
            mybounds[GABOR_PARAM_T][0] = s * t0_scl;
            mybounds[GABOR_PARAM_T][1] = s * t1_scl;
            mybounds[GABOR_PARAM_L][0] = s * l0_scl;
            mybounds[GABOR_PARAM_L][1] = s * l1_scl;
        }
        
    }
    
}

//////////////////////////////////////////////////////////////////////

const float* magma_data();

GLuint setup_textures() {

    make_texture(&src_image32f);

    make_texture(&weight_image32f);

    memset(&palette_image32f, 0, sizeof(palette_image32f));
    
    palette_image32f.width = 256;
    palette_image32f.height = 1;
    palette_image32f.channels = 3;
    palette_image32f.num_elements = 256*3;
    palette_image32f.type = IMAGE_32F;
    palette_image32f.element_size = sizeof(float);
    palette_image32f.data_32f = (float*)magma_data();
    
    palette_image32f.buf.size = palette_image32f.num_elements * palette_image32f.element_size;
    palette_image32f.buf.data = (char*)palette_image32f.data_32f;
    palette_image32f.buf.alloc = palette_image32f.buf.size;

    make_texture(&palette_image32f);
    

    //////////////////////////////////////////////////////////////////////
    // setup param texture

    float hwmax = MAX(src_image32f.width, src_image32f.height);
    px = 2.0 / hwmax;

    param_bounds[GABOR_PARAM_S][0] = 0.5*px;
    
    image_create(&param_image32f, gabors_per_tile*3, num_tiles, 4, IMAGE_32F);

    for (int pidx=0; pidx<num_tiles; ++pidx) {
        for (int midx=0; midx<gabors_per_tile; ++midx) {
            int i = pidx * gabors_per_tile + midx;
            float* pi = param_image32f.data_32f + i*GABOR_NUM_PARAMS;
            init_params(pi);
        }
    }

    return make_texture(&param_image32f);

}

//////////////////////////////////////////////////////////////////////

void setup_vertex_stuff() {

    vertex_shader = make_shader_file(GL_VERTEX_SHADER, "../vertex.glsl", 1024);

    const GLfloat vertices[4][2] = {
        { -1.f, -1.f  },
        {  1.f, -1.f  },
        {  1.f,  1.f  },
        { -1.f,  1.f  }
    };

    GLubyte indices[] = { 0, 1, 2, 0, 2, 3 };

    GLuint vertex_buffer, element_buffer, vao;

    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices),
                 vertices, GL_STATIC_DRAW);
    
    check_opengl_errors("after vertex buffer setup");

    glGenBuffers(1, &element_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices),
                 indices, GL_STATIC_DRAW);
    
    check_opengl_errors("after element buffer setup");

    glGenVertexArrays(1, &vao);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer);
    
    check_opengl_errors("after vao setup");

}

//////////////////////////////////////////////////////////////////////

void setup_framebuffers(GLFWwindow* window) {

    snprintf(common_defines, sizeof(common_defines),
             "#define GABORS_PER_TILE %d\n"
             "#define SUM_WINDOW_SIZE %d\n",
             (int)gabors_per_tile, SUM_WINDOW_SIZE);

    printf("common_defines is:\n%s", common_defines);

    ////////////////////////////////////////////////////////////
    
    fb_setup(&gabor_eval_fb,
             "gabor_eval",
             src_image32f.width,
             src_image32f.height * num_tiles,
             GL_RGBA32F,
             "../gabor_eval.glsl", MAX_SOURCE_LENGTH);

    fb_add_input(&gabor_eval_fb, "paramTexture", param_image32f.bound_texture);

    float fdims[2] = { src_image32f.width, src_image32f.height };
    
    fb_enqueue_uupdate_f(&gabor_eval_fb, "srcDims", 1, fdims, glUniform2fv);

    fb_run_uupdates(&gabor_eval_fb);
    
    ////////////////////////////////////////////////////////////
    
    fb_setup(&gabor_compare_fb,
             "gabor_compare",
             src_image32f.width,
             src_image32f.height * num_tiles,
             GL_RGBA32F,
             "../gabor_compare.glsl", MAX_SOURCE_LENGTH);

    fb_add_input(&gabor_compare_fb, "approxTexture",
                 gabor_eval_fb.render_texture);

    fb_add_input(&gabor_compare_fb, "srcTexture",
                 src_image32f.bound_texture);

    fb_add_input(&gabor_compare_fb, "weightTexture",
                 weight_image32f.bound_texture);
    
    ////////////////////////////////////////////////////////////
    // set up reduce buffers
    
    num_reduce = 0;
    float w = src_image32f.width, h = src_image32f.height;
    
    while (w > 1 || h > 1) {
        w = ceil( w / SUM_WINDOW_SIZE );
        h = ceil( h / SUM_WINDOW_SIZE );
        num_reduce += 1;
    }

    reduce_fbs = calloc(num_reduce, sizeof(framebuffer_t));
    reduce_names = calloc(num_reduce, 32);
    
    w = src_image32f.width;
    h = src_image32f.height;

    GLuint input_texture = gabor_compare_fb.render_texture;

    for (size_t i=0; i<num_reduce; ++i) {

        float prev_w = w;
        float prev_h = h;
        
        w = ceil( w / SUM_WINDOW_SIZE );
        h = ceil( h / SUM_WINDOW_SIZE );

        char* reduce_name = reduce_names + 32*i;

        snprintf(reduce_name, 32, "reduce%d", (int)i);
        
        fb_setup(reduce_fbs + i, reduce_name,
                 w, h * num_tiles, GL_RGBA32F,
                 "../reduce.glsl", MAX_SOURCE_LENGTH);

        fb_add_input(reduce_fbs + i,
                     "inputTexture",
                     input_texture);

        int idims[2] = { prev_w, prev_h };
        int odims[2] = { w, h };
        
        fb_enqueue_uupdate_i(reduce_fbs + i,
                             "inputDims", 1,
                             idims, glUniform2iv);

        fb_enqueue_uupdate_i(reduce_fbs + i,
                             "outputDims", 1,
                             odims, glUniform2iv);

        fb_run_uupdates(reduce_fbs + i);
        
        input_texture = reduce_fbs[i].render_texture;

    }

    image_create(&reduced_image32f, 1, num_tiles, 4, IMAGE_32F);
    printf("reduced_image32f has size %d\n", (int)reduced_image32f.buf.size);

    objective_values = malloc(num_tiles*sizeof(float));

    fb_setup(&main_fb,
             "main",
             VIS_TILES*src_image32f.width,
             src_image32f.height,
             GL_RGBA32F,
             "../visualize.glsl", MAX_SOURCE_LENGTH);

    fb_add_input(&main_fb, "srcTexture", src_image32f.bound_texture);
    fb_add_input(&main_fb, "approxTexture", gabor_eval_fb.render_texture);
    fb_add_input(&main_fb, "errorTexture", gabor_compare_fb.render_texture);
    fb_add_input(&main_fb, "palette", palette_image32f.bound_texture);

    int sdims[2] = { src_image32f.width, src_image32f.height };
    int ntiles[2] = { VIS_TILES, num_tiles };

    fb_enqueue_uupdate_i(&main_fb, "srcDims", 1, sdims, glUniform2iv);

    fb_enqueue_uupdate_i(&main_fb, "numTiles", 1, ntiles, glUniform2iv);

    fb_run_uupdates(&main_fb);

}

//////////////////////////////////////////////////////////////////////

void compute() {

    upload_texture(&param_image32f);

    fb_draw(&gabor_eval_fb);
    fb_draw(&gabor_compare_fb);

    for (size_t i=0; i<num_reduce; ++i) {
        fb_draw(reduce_fbs + i);
    }
    
    read_pixels(&reduced_image32f);

    for (size_t i=0; i<num_tiles; ++i) {
        float num = reduced_image32f.data_32f[4*i + 0];
        float denom = reduced_image32f.data_32f[4*i + 3];
        objective_values[i] = num / denom;
    }

}

//////////////////////////////////////////////////////////////////////

void anneal_init() {

    anneal.prev_cost = -1;

    anneal.t_max = 5e-5;
    
    const double max_iter = 1e8;
    const double temp_decay = 1e-4;
    anneal.t_rate = -log(temp_decay) / max_iter;

    anneal.change_fraction = 0.0;
    anneal.p_reinitialize = 0.01;
    anneal.mutate_amount = 0.01;

    image_copy(&anneal.good_params32f, &param_image32f);

}

//////////////////////////////////////////////////////////////////////

void anneal_update(size_t iteration) {

    float cur_cost = objective_values[0];

    int first = (iteration == 0);


    if (first) {
        anneal.prev_cost = cur_cost;
        return;
    }

    int keep = 0;

    double delta_cost = anneal.prev_cost - cur_cost;

    if (delta_cost > 0) {
        keep = 1;
    } else {
        double temperature = anneal.t_max * exp(-anneal.t_rate * iteration);
        float p_keep = exp(delta_cost / temperature);
        if (random_float() < p_keep) {
            keep = 1;
        }
    }

    if (keep) {
        
        anneal.prev_cost = cur_cost;

        memcpy(anneal.good_params32f.buf.data,
               (const char*)param_image32f.buf.data,
               param_image32f.buf.size);
        
    } else {
        
        memcpy(param_image32f.buf.data,
               (const char*)anneal.good_params32f.buf.data,
               param_image32f.buf.size);
        
    }


    // tweak multiple gabor functions
    size_t nchange = floor(anneal.change_fraction * gabors_per_tile + 0.5);
    if (nchange < 1) { nchange = 1; }
        

    require( num_tiles == 1 );

    for (size_t pidx=0; pidx<num_tiles; ++pidx) {
        for (size_t c=0; c<nchange; ++c) {

            int midx = pcg32_random_r(&rng_global) % gabors_per_tile;
            int i = midx + pidx * gabors_per_tile;
        
            float* pi = param_image32f.data_32f + i*GABOR_NUM_PARAMS;
        
            float r = random_float();
        
            if (r < anneal.p_reinitialize) {
                init_params(pi);
            } else {
                mutate_params(pi, anneal.mutate_amount);
            }

        }

    }
}

//////////////////////////////////////////////////////////////////////

void anneal_info(size_t iteration,
                 double elapsed,
                 size_t num_iter) {

    double temperature = anneal.t_max * exp(-anneal.t_rate * iteration);
    
    printf("ran %zu iterations in %g seconds (%g ms/iter); "
           "at iteration %zu, cost is %g and temperature is %g\n",
           num_iter, elapsed, 1000*elapsed/num_iter,
           iteration, objective_values[0], temperature);

    
}

//////////////////////////////////////////////////////////////////////

void solve(GLFWwindow* window) {

    require(solver == SOLVER_ANNEALING);
    anneal_init();

    double start = glfwGetTime();
    
    size_t iteration = 0;
    size_t iter_since_printout = 0;

    const size_t iter_per_vis = 1000;
    const size_t iter_per_screenshot = 1000;


    int num_screenshots = 0;

    while (1) {

        compute();
        
        int do_vis = iter_per_vis && (iteration % iter_per_vis == 0);
        int do_screenshot = iter_per_screenshot && (iteration % iter_per_screenshot == 0);

        if (do_vis || do_screenshot) {
 
            double elapsed = glfwGetTime() - start;
            anneal_info(iteration, elapsed, iter_since_printout);
                   
            if (do_vis) {
                draw_main(window);
                glfwSwapBuffers(window);
                glfwPollEvents();
                if (glfwWindowShouldClose(window)) { break; }
            }

            if (do_screenshot) {
                main_fb.request_screenshot = num_screenshots++;
                draw_main(NULL);
            }

            start = glfwGetTime();
            iter_since_printout = 0;

        }
        
        anneal_update(iteration);

        ++iteration;
        ++iter_since_printout;
        
    }

}
    
//////////////////////////////////////////////////////////////////////

const float* magma_data() {

    // from https://github.com/BIDS/colormap/blob/master/colormaps.py
    static const float data[256*3] = {
        0.001462, 0.000466, 0.013866,
        0.002258, 0.001295, 0.018331,
        0.003279, 0.002305, 0.023708,
        0.004512, 0.003490, 0.029965,
        0.005950, 0.004843, 0.037130,
        0.007588, 0.006356, 0.044973,
        0.009426, 0.008022, 0.052844,
        0.011465, 0.009828, 0.060750,
        0.013708, 0.011771, 0.068667,
        0.016156, 0.013840, 0.076603,
        0.018815, 0.016026, 0.084584,
        0.021692, 0.018320, 0.092610,
        0.024792, 0.020715, 0.100676,
        0.028123, 0.023201, 0.108787,
        0.031696, 0.025765, 0.116965,
        0.035520, 0.028397, 0.125209,
        0.039608, 0.031090, 0.133515,
        0.043830, 0.033830, 0.141886,
        0.048062, 0.036607, 0.150327,
        0.052320, 0.039407, 0.158841,
        0.056615, 0.042160, 0.167446,
        0.060949, 0.044794, 0.176129,
        0.065330, 0.047318, 0.184892,
        0.069764, 0.049726, 0.193735,
        0.074257, 0.052017, 0.202660,
        0.078815, 0.054184, 0.211667,
        0.083446, 0.056225, 0.220755,
        0.088155, 0.058133, 0.229922,
        0.092949, 0.059904, 0.239164,
        0.097833, 0.061531, 0.248477,
        0.102815, 0.063010, 0.257854,
        0.107899, 0.064335, 0.267289,
        0.113094, 0.065492, 0.276784,
        0.118405, 0.066479, 0.286321,
        0.123833, 0.067295, 0.295879,
        0.129380, 0.067935, 0.305443,
        0.135053, 0.068391, 0.315000,
        0.140858, 0.068654, 0.324538,
        0.146785, 0.068738, 0.334011,
        0.152839, 0.068637, 0.343404,
        0.159018, 0.068354, 0.352688,
        0.165308, 0.067911, 0.361816,
        0.171713, 0.067305, 0.370771,
        0.178212, 0.066576, 0.379497,
        0.184801, 0.065732, 0.387973,
        0.191460, 0.064818, 0.396152,
        0.198177, 0.063862, 0.404009,
        0.204935, 0.062907, 0.411514,
        0.211718, 0.061992, 0.418647,
        0.218512, 0.061158, 0.425392,
        0.225302, 0.060445, 0.431742,
        0.232077, 0.059889, 0.437695,
        0.238826, 0.059517, 0.443256,
        0.245543, 0.059352, 0.448436,
        0.252220, 0.059415, 0.453248,
        0.258857, 0.059706, 0.457710,
        0.265447, 0.060237, 0.461840,
        0.271994, 0.060994, 0.465660,
        0.278493, 0.061978, 0.469190,
        0.284951, 0.063168, 0.472451,
        0.291366, 0.064553, 0.475462,
        0.297740, 0.066117, 0.478243,
        0.304081, 0.067835, 0.480812,
        0.310382, 0.069702, 0.483186,
        0.316654, 0.071690, 0.485380,
        0.322899, 0.073782, 0.487408,
        0.329114, 0.075972, 0.489287,
        0.335308, 0.078236, 0.491024,
        0.341482, 0.080564, 0.492631,
        0.347636, 0.082946, 0.494121,
        0.353773, 0.085373, 0.495501,
        0.359898, 0.087831, 0.496778,
        0.366012, 0.090314, 0.497960,
        0.372116, 0.092816, 0.499053,
        0.378211, 0.095332, 0.500067,
        0.384299, 0.097855, 0.501002,
        0.390384, 0.100379, 0.501864,
        0.396467, 0.102902, 0.502658,
        0.402548, 0.105420, 0.503386,
        0.408629, 0.107930, 0.504052,
        0.414709, 0.110431, 0.504662,
        0.420791, 0.112920, 0.505215,
        0.426877, 0.115395, 0.505714,
        0.432967, 0.117855, 0.506160,
        0.439062, 0.120298, 0.506555,
        0.445163, 0.122724, 0.506901,
        0.451271, 0.125132, 0.507198,
        0.457386, 0.127522, 0.507448,
        0.463508, 0.129893, 0.507652,
        0.469640, 0.132245, 0.507809,
        0.475780, 0.134577, 0.507921,
        0.481929, 0.136891, 0.507989,
        0.488088, 0.139186, 0.508011,
        0.494258, 0.141462, 0.507988,
        0.500438, 0.143719, 0.507920,
        0.506629, 0.145958, 0.507806,
        0.512831, 0.148179, 0.507648,
        0.519045, 0.150383, 0.507443,
        0.525270, 0.152569, 0.507192,
        0.531507, 0.154739, 0.506895,
        0.537755, 0.156894, 0.506551,
        0.544015, 0.159033, 0.506159,
        0.550287, 0.161158, 0.505719,
        0.556571, 0.163269, 0.505230,
        0.562866, 0.165368, 0.504692,
        0.569172, 0.167454, 0.504105,
        0.575490, 0.169530, 0.503466,
        0.581819, 0.171596, 0.502777,
        0.588158, 0.173652, 0.502035,
        0.594508, 0.175701, 0.501241,
        0.600868, 0.177743, 0.500394,
        0.607238, 0.179779, 0.499492,
        0.613617, 0.181811, 0.498536,
        0.620005, 0.183840, 0.497524,
        0.626401, 0.185867, 0.496456,
        0.632805, 0.187893, 0.495332,
        0.639216, 0.189921, 0.494150,
        0.645633, 0.191952, 0.492910,
        0.652056, 0.193986, 0.491611,
        0.658483, 0.196027, 0.490253,
        0.664915, 0.198075, 0.488836,
        0.671349, 0.200133, 0.487358,
        0.677786, 0.202203, 0.485819,
        0.684224, 0.204286, 0.484219,
        0.690661, 0.206384, 0.482558,
        0.697098, 0.208501, 0.480835,
        0.703532, 0.210638, 0.479049,
        0.709962, 0.212797, 0.477201,
        0.716387, 0.214982, 0.475290,
        0.722805, 0.217194, 0.473316,
        0.729216, 0.219437, 0.471279,
        0.735616, 0.221713, 0.469180,
        0.742004, 0.224025, 0.467018,
        0.748378, 0.226377, 0.464794,
        0.754737, 0.228772, 0.462509,
        0.761077, 0.231214, 0.460162,
        0.767398, 0.233705, 0.457755,
        0.773695, 0.236249, 0.455289,
        0.779968, 0.238851, 0.452765,
        0.786212, 0.241514, 0.450184,
        0.792427, 0.244242, 0.447543,
        0.798608, 0.247040, 0.444848,
        0.804752, 0.249911, 0.442102,
        0.810855, 0.252861, 0.439305,
        0.816914, 0.255895, 0.436461,
        0.822926, 0.259016, 0.433573,
        0.828886, 0.262229, 0.430644,
        0.834791, 0.265540, 0.427671,
        0.840636, 0.268953, 0.424666,
        0.846416, 0.272473, 0.421631,
        0.852126, 0.276106, 0.418573,
        0.857763, 0.279857, 0.415496,
        0.863320, 0.283729, 0.412403,
        0.868793, 0.287728, 0.409303,
        0.874176, 0.291859, 0.406205,
        0.879464, 0.296125, 0.403118,
        0.884651, 0.300530, 0.400047,
        0.889731, 0.305079, 0.397002,
        0.894700, 0.309773, 0.393995,
        0.899552, 0.314616, 0.391037,
        0.904281, 0.319610, 0.388137,
        0.908884, 0.324755, 0.385308,
        0.913354, 0.330052, 0.382563,
        0.917689, 0.335500, 0.379915,
        0.921884, 0.341098, 0.377376,
        0.925937, 0.346844, 0.374959,
        0.929845, 0.352734, 0.372677,
        0.933606, 0.358764, 0.370541,
        0.937221, 0.364929, 0.368567,
        0.940687, 0.371224, 0.366762,
        0.944006, 0.377643, 0.365136,
        0.947180, 0.384178, 0.363701,
        0.950210, 0.390820, 0.362468,
        0.953099, 0.397563, 0.361438,
        0.955849, 0.404400, 0.360619,
        0.958464, 0.411324, 0.360014,
        0.960949, 0.418323, 0.359630,
        0.963310, 0.425390, 0.359469,
        0.965549, 0.432519, 0.359529,
        0.967671, 0.439703, 0.359810,
        0.969680, 0.446936, 0.360311,
        0.971582, 0.454210, 0.361030,
        0.973381, 0.461520, 0.361965,
        0.975082, 0.468861, 0.363111,
        0.976690, 0.476226, 0.364466,
        0.978210, 0.483612, 0.366025,
        0.979645, 0.491014, 0.367783,
        0.981000, 0.498428, 0.369734,
        0.982279, 0.505851, 0.371874,
        0.983485, 0.513280, 0.374198,
        0.984622, 0.520713, 0.376698,
        0.985693, 0.528148, 0.379371,
        0.986700, 0.535582, 0.382210,
        0.987646, 0.543015, 0.385210,
        0.988533, 0.550446, 0.388365,
        0.989363, 0.557873, 0.391671,
        0.990138, 0.565296, 0.395122,
        0.990871, 0.572706, 0.398714,
        0.991558, 0.580107, 0.402441,
        0.992196, 0.587502, 0.406299,
        0.992785, 0.594891, 0.410283,
        0.993326, 0.602275, 0.414390,
        0.993834, 0.609644, 0.418613,
        0.994309, 0.616999, 0.422950,
        0.994738, 0.624350, 0.427397,
        0.995122, 0.631696, 0.431951,
        0.995480, 0.639027, 0.436607,
        0.995810, 0.646344, 0.441361,
        0.996096, 0.653659, 0.446213,
        0.996341, 0.660969, 0.451160,
        0.996580, 0.668256, 0.456192,
        0.996775, 0.675541, 0.461314,
        0.996925, 0.682828, 0.466526,
        0.997077, 0.690088, 0.471811,
        0.997186, 0.697349, 0.477182,
        0.997254, 0.704611, 0.482635,
        0.997325, 0.711848, 0.488154,
        0.997351, 0.719089, 0.493755,
        0.997351, 0.726324, 0.499428,
        0.997341, 0.733545, 0.505167,
        0.997285, 0.740772, 0.510983,
        0.997228, 0.747981, 0.516859,
        0.997138, 0.755190, 0.522806,
        0.997019, 0.762398, 0.528821,
        0.996898, 0.769591, 0.534892,
        0.996727, 0.776795, 0.541039,
        0.996571, 0.783977, 0.547233,
        0.996369, 0.791167, 0.553499,
        0.996162, 0.798348, 0.559820,
        0.995932, 0.805527, 0.566202,
        0.995680, 0.812706, 0.572645,
        0.995424, 0.819875, 0.579140,
        0.995131, 0.827052, 0.585701,
        0.994851, 0.834213, 0.592307,
        0.994524, 0.841387, 0.598983,
        0.994222, 0.848540, 0.605696,
        0.993866, 0.855711, 0.612482,
        0.993545, 0.862859, 0.619299,
        0.993170, 0.870024, 0.626189,
        0.992831, 0.877168, 0.633109,
        0.992440, 0.884330, 0.640099,
        0.992089, 0.891470, 0.647116,
        0.991688, 0.898627, 0.654202,
        0.991332, 0.905763, 0.661309,
        0.990930, 0.912915, 0.668481,
        0.990570, 0.920049, 0.675675,
        0.990175, 0.927196, 0.682926,
        0.989815, 0.934329, 0.690198,
        0.989434, 0.941470, 0.697519,
        0.989077, 0.948604, 0.704863,
        0.988717, 0.955742, 0.712242,
        0.988367, 0.962878, 0.719649,
        0.988033, 0.970012, 0.727077,
        0.987691, 0.977154, 0.734536,
        0.987387, 0.984288, 0.742002,
        0.987053, 0.991438, 0.749504
    };

    return data;

}

//////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {


    struct timeval tv;
    gettimeofday(&tv, NULL);

    pcg32_srandom_r(&rng_global, tv.tv_sec, tv.tv_usec);

    get_options(argc, argv);    

    GLFWwindow* window = setup_window();


    setup_textures();
             
    setup_vertex_stuff();

    setup_framebuffers(window);

    solve(window);

    
}

