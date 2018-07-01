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

 - add weight texture
 - resize input using mipmapping
 - nice palette (magma?) for output

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
    MAX_INPUTS = 4,
};

typedef struct framebuffer {

    const char* name;

    size_t width, height;
    GLenum internal_format;

    GLuint render_texture;
    GLuint framebuffer;

    size_t num_inputs;
    GLenum inputs[MAX_INPUTS];

    GLuint program;

    int request_screenshot;

} framebuffer_t;
    

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

image_t src_image8u;
image_t src_image32f;

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

typedef struct anneal_info {
    int iteration;
    float prev_cost;
    double t_max;
    double t_rate;
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

void image_copy(image_t* dst, const image_t* src) {

    image_create(dst, src->width, src->height, src->channels, src->type);

    require(dst->buf.size == src->buf.size);

    memcpy(dst->buf.data, src->buf.data, src->buf.size);

}


void image8u_to_32f(const image_t* src,
                    image_t* dst) {

    require(src->type == IMAGE_8U);

    image_create(dst, src->width, src->height, src->channels, IMAGE_32F);

    for (size_t i=0; i<src->num_elements; ++i) {
        dst->data_32f[i] = (float)src->data_8u[i] / 255.0f;
    }

}

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

    fprintf(stderr, "wrote %s with size %dx%d\n", filename,
            (int)ncols, (int)nrows);

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
    
    if (color_type == PNG_COLOR_TYPE_GRAY ) {
        channels = 3;
        png_set_gray_to_rgb(png_ptr);
    } else if (color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        channels = 4;
        png_set_gray_to_rgb(png_ptr);
    } else if (color_type == PNG_COLOR_TYPE_PALETTE) {
        channels = 3;
        png_set_palette_to_rgb(png_ptr);
        if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
            channels = 4;
            png_set_tRNS_to_alpha(png_ptr);
        }
    }
    
    if (width <= 0 || height <= 0 || bitdepth != 8 ||
        (channels != 3 && channels != 4)) {
        
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


void load_image(image_t* dst, const char* filename) {

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
        read_png(&tmp, 0, dst);
    } else {
        read_jpg(&tmp, 0, dst);
    }

    buf_free(&tmp);
    
}

//////////////////////////////////////////////////////////////////////

void dieusage() {

    fprintf(stderr,
            "usage: rgb_imfit INPUTIMAGE\n");

    exit(1);

}

//////////////////////////////////////////////////////////////////////

void get_options(int argc, char** argv) {

    if (argc != 2) {
        dieusage();
    }

    const char* filename = argv[argc-1];
    
    load_image(&src_image8u, filename);
    
    printf("converting...\n");
    image8u_to_32f(&src_image8u, &src_image32f);
    printf("converted!\n");

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

GLenum gl_format(size_t channels) {
    require( channels == 3 || channels == 4 );
    if (channels == 3) {
        return GL_RGB;
    } else {
        return GL_RGBA;
    }
}

GLenum gl_internal_format(GLenum format, GLenum datatype) {

    require(format == GL_RGB || format == GL_RGBA);
    require(datatype == GL_UNSIGNED_BYTE || datatype == GL_FLOAT);

    if (format == GL_RGB) {
        return datatype == GL_FLOAT ? GL_RGB32F : GL_RGB8;
    } else {
        return datatype == GL_FLOAT ? GL_RGBA32F : GL_RGBA8;
    }
    
}

//////////////////////////////////////////////////////////////////////

// should already be bound!
void upload_texture(const image_t* image) {

    GLenum format = gl_format(image->channels);
    GLenum datatype = gl_datatype(image->type);

    glBindTexture(GL_TEXTURE_2D, image->bound_texture);
    
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    image->width, image->height,
                    format, datatype,
                    image->buf.data);


}

GLuint make_texture(image_t* image) {

    GLuint texname;
    
    glGenTextures(1, &texname);
    glBindTexture(GL_TEXTURE_2D, texname);

    
    GLenum format = gl_format(image->channels);
    GLenum datatype = gl_datatype(image->type);
    GLenum internal_format = gl_internal_format(format, datatype);
        
    glTexStorage2D(GL_TEXTURE_2D, 1, internal_format,
                   image->width, image->height);

    image->bound_texture = texname;
    upload_texture(image);
    
    check_opengl_errors("make texture32f!");

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

    int w = 3*src_image32f.width, h = src_image32f.height;

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
    
    GLint src_dims = glGetUniformLocation(program, "srcDims");
    if (src_dims != -1) {
        float dims[2] = { src_image32f.width, src_image32f.height };
        printf("  found srcDims uniform, setting to %f %f!\n",
               dims[0], dims[1]);
        glUniform2fv(src_dims, 1, dims);
    }

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
    snprintf(filename, 1024, "%s.png", fb->name);
  
    write_png(filename, screen, w, h, stride, 0, NULL);

    free(screen);

  
}

//////////////////////////////////////////////////////////////////////

void fb_draw(framebuffer_t* fb) {

    glBindFramebuffer(GL_FRAMEBUFFER, fb->framebuffer);
    glViewport(0, 0, fb->width, fb->height);
    glUseProgram(fb->program);

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
        printf("************************************************************\n");
        fb_screenshot(fb);
        fb->request_screenshot = 0;
        printf("************************************************************\n");
    }
    
}


//////////////////////////////////////////////////////////////////////

typedef struct { uint64_t state;  uint64_t inc; } pcg32_random_t;

#define PCG32_INITIALIZER   { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL }

pcg32_random_t rng_global = PCG32_INITIALIZER;

uint32_t pcg32_random_r(pcg32_random_t* rng) {
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
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

float random_float() {
    return (float)pcg32_random_r(&rng_global) / (float)UINT32_MAX;
}

void random_float_sign(float* x, float* s) {
    uint32_t r = pcg32_random_r(&rng_global);
    *s = (r & 1) ? -1 : 1;
    *x = (float)(r >> 1) / (float)(UINT32_MAX >> 1);
}

float lerp(float a, float b, float u) {
    return a + u*(b-a);
}

#define MAX(a,b) ((a) > (b) ? (a) : (b))

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

float clamp(float x, float minval, float maxval) {
    return x < minval ? minval : x > maxval ? maxval : x;
}

float wrap2pi(float x) {
    return fmod(x, 2*M_PI);
}

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

GLuint setup_param_texture() {

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
            /*
            if (pidx == 0) {
                init_params(pi);
            } else {
                const float* pi0 = param_image32f.data_32f + midx*GABOR_NUM_PARAMS;
                require(pi0 < pi);
                memcpy(pi, pi0, sizeof(float)*GABOR_NUM_PARAMS);
                mutate_params(pi, 0.05);
            }
            */
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

        GLint input_dims = glGetUniformLocation(reduce_fbs[i].program,
                                                "inputDims");

        require( input_dims != -1 );

        int idims[2] = { prev_w, prev_h };
        int odims[2] = { w, h };

        glUniform2iv(input_dims, 1, idims);

        printf("  set input_dims to %d, %d\n", idims[0], idims[1]);

        GLint output_dims = glGetUniformLocation(reduce_fbs[i].program,
                                                "outputDims");

        require( output_dims != -1 );

        glUniform2iv(output_dims, 1, odims);

        printf("  set output_dims to %d, %d\n", odims[0], odims[1]);
        
        input_texture = reduce_fbs[i].render_texture;

    }

    image_create(&reduced_image32f, 1, num_tiles, 4, IMAGE_32F);
    printf("reduced_image32f has size %d\n", (int)reduced_image32f.buf.size);

    int mw, mh;
    
    glfwGetFramebufferSize(window, &mw, &mh);

    fb_setup(&main_fb,
             "main",
             mw, mh,
             GL_NONE,
             "../visualize.glsl", MAX_SOURCE_LENGTH);

    fb_add_input(&main_fb, "srcTexture", src_image32f.bound_texture);
    fb_add_input(&main_fb, "approxTexture", gabor_eval_fb.render_texture);
    fb_add_input(&main_fb, "errorTexture", gabor_compare_fb.render_texture);

}

void compute() {

    // don't forget inequalities:
    //  s >= l/32, s < l/2
    //  t > s,     t < 8s (basically a pyramid in 3D)

    upload_texture(&param_image32f);

    fb_draw(&gabor_eval_fb);
    fb_draw(&gabor_compare_fb);

    for (size_t i=0; i<num_reduce; ++i) {
        fb_draw(reduce_fbs + i);
    }
    
    read_pixels(&reduced_image32f);

}

void annealing_init() {

    require(num_tiles == 1);

    anneal.iteration = 0;
    anneal.prev_cost = -1;

    // p_init = exp(delta_init / T)
    // log( p_init ) = delta_init / T
    // T = delta_init / log(p_init)
    
    const double p_init = 0.01;
    const double delta_init = -0.0001;
    anneal.t_max = delta_init / log(p_init);
    
    // exp(-max_iter*t_rate) = temp_decrease
    // -max_iter*t_rate = log(temp_decrease)
    // t_rate = -log(temp_decrease)/max_iter

    const double max_iter = 1e7; // 10 million
    const double temp_decrease = 1e-4;
    anneal.t_rate = -log(temp_decrease) / max_iter;

    printf("p_init = %g, exp(delta_init / T) = %g\n",
           p_init, exp(delta_init / anneal.t_max));

    printf("t_max = %g\n", anneal.t_max);

    anneal.p_reinitialize = 0.01;
    anneal.mutate_amount = 0.01;

    image_copy(&anneal.good_params32f, &param_image32f);

}

void annealing_move() {
    
    if (anneal.iteration == 0) {
        return;
    }

    require(num_tiles == 1);

    // tweak exactly one gabor function
    int midx = pcg32_random_r(&rng_global) % gabors_per_tile;
    int i = midx * GABOR_NUM_PARAMS;

    float* pi = param_image32f.data_32f + i;
    
    float r = random_float();

    if (r < anneal.p_reinitialize) {
        init_params(pi);
    } else {
        mutate_params(pi, anneal.mutate_amount);
    }
    
}

void annealing_verify() {

    float num = reduced_image32f.data_32f[0];
    float denom = reduced_image32f.data_32f[3];
    float cur_cost = num / denom;

    //printf("num = %g, denom = %g\n", num, denom);
    //printf("cur_cost at iteration %d = %g (%g prev)\n",
    //anneal.iteration, cur_cost, anneal.prev_cost);
    
    int first = (anneal.iteration == 0);
    ++anneal.iteration;

    if (first) {
        anneal.prev_cost = cur_cost;
        return;
    }

    int keep = 0;

    double delta_cost = anneal.prev_cost - cur_cost;

    if (delta_cost > 0) {
        //printf("improved, keeping it!\n");
        keep = 1;
    } else {
        double temperature = anneal.t_max * exp(-anneal.t_rate * anneal.iteration);
        float p_keep = exp(delta_cost / temperature);
        //printf("for delta_cost=%g, p_keep=%g with temp=%g\n",
        //delta_cost, p_keep, temperature);
        if (random_float() < p_keep) {
            //printf("accepted!\n");
            keep = 1;
        }
    }

    if (keep) {
        
        //printf("updating cost!\n");
        anneal.prev_cost = cur_cost;

        memcpy(anneal.good_params32f.buf.data,
               (const char*)param_image32f.buf.data,
               param_image32f.buf.size);
        
    } else {
        
        //printf("reverting texture!\n");

        memcpy(param_image32f.buf.data,
               (const char*)anneal.good_params32f.buf.data,
               param_image32f.buf.size);
    }

    
}

void anneal_info(double elapsed, int num_iter) {

    double temperature = anneal.t_max * exp(-anneal.t_rate * anneal.iteration);
    
    printf("ran %d iterations in %g seconds (%g ms/iter); "
           "at iteration %d, cost is %g and temperature is %g\n",
           num_iter, elapsed, 1000*elapsed/num_iter,
           anneal.iteration, anneal.prev_cost, temperature);

}

void solve(GLFWwindow* window) {

    require(solver == SOLVER_ANNEALING);
    annealing_init();

    int done = 0;

    while (!done) {

        memcpy(param_image32f.buf.data,
               (const char*)anneal.good_params32f.buf.data,
               param_image32f.buf.size);
        
        compute();

        int w, h;
        glfwGetFramebufferSize(window, &w, &h);

        main_fb.width = w;
        main_fb.height = h;

        fb_draw(&main_fb);

        glfwSwapBuffers(window);
        glfwPollEvents();

        double start = glfwGetTime();
        const int iter_per_update = 5000;
        
        for (int i=0; i<iter_per_update; ++i) {

            if (glfwWindowShouldClose(window)) {
                done = 1;
                break;
            }

            annealing_move();
            
            compute();
            
            annealing_verify();

        }

        double elapsed = glfwGetTime() - start;

        anneal_info(elapsed, iter_per_update);

        /*
        double start = glfwGetTime();

        for (size_t i=0; i<num_profile; ++i) {

            compute(do_screenshots);
            if (do_screenshots) { break; }
            
        }

        glFinish();

        double elapsed = glfwGetTime() - start;

        printf("ran %d frames in %.4f seconds (%.4f ms/frame)\n",
               (int)num_profile, elapsed,
               1000*elapsed/num_profile);

        
        float image_size = src_image32f.width * src_image32f.height;

        for (size_t i=0; i<num_tiles; ++i) {
            size_t offs = 4*i;
            //printf("reduced_image32f[%d].a = %f\n", (int)i, reduced_image32f.data_32f[offs+3]);
            require( reduced_image32f.data_32f[offs + 3] == image_size );
        }
        
        return 0;
        
        printf("foo!\n");
        glfwPollEvents();
        */
        
    }

}
    
//////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {


    struct timeval tv;
    gettimeofday(&tv, NULL);

    pcg32_srandom_r(&rng_global, tv.tv_sec, tv.tv_usec);

    get_options(argc, argv);    

    GLFWwindow* window = setup_window();

    make_texture(&src_image32f);

    setup_param_texture();
             
    setup_vertex_stuff();

    setup_framebuffers(window);

    solve(window);

    
}
