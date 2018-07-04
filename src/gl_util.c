#include "gl_util.h"
#include "require.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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

GLenum gl_datatype(image_type_t type) {
    require( type == IMAGE_8U || type == IMAGE_32F || type == IMAGE_32U );
    if (type == IMAGE_8U) {
        return GL_UNSIGNED_BYTE;
    } else if (type == IMAGE_32F) {
        return GL_FLOAT;
    } else {
        return GL_UNSIGNED_INT;
    }

}

//////////////////////////////////////////////////////////////////////

GLenum gl_format(size_t channels,
                 image_type_t datatype) {
    
    require(channels >= 1 && channels <= 4 );
    require( channels >= 1 && channels <= 4 );

    static const GLenum tbl[3][4] = {
        { GL_RED,  GL_RG,  GL_RGB, GL_RGBA },
        { GL_RED,  GL_RG,  GL_RGB, GL_RGBA },
        { GL_RED_INTEGER, GL_RG_INTEGER, GL_RGB_INTEGER, GL_RGBA_INTEGER }
    };

    return tbl[datatype][channels-1];
}

//////////////////////////////////////////////////////////////////////

GLenum gl_internal_format(size_t channels,
                          image_type_t datatype) {

    require(channels >= 1 && channels <= 4 );
    require((int)datatype >= 0 && (int)datatype < 3);

    static const GLenum tbl[3][4] = {
        { GL_R8,    GL_RG8,    GL_RGB8,    GL_RGBA8 },
        { GL_R32F,  GL_RG32F,  GL_RGB32F,  GL_RGBA32F },
        { GL_R32UI, GL_RG32UI, GL_RGB32UI, GL_RGBA32UI }
    };

    return tbl[datatype][channels-1];
    
}

//////////////////////////////////////////////////////////////////////

void upload_texture(const image_t* image) {

    GLenum format = gl_format(image->channels, image->type);
    GLenum datatype = gl_datatype(image->type);

    glBindTexture(GL_TEXTURE_2D, image->bound_texture);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    image->width, image->height,
                    format, datatype,
                    image->buf.data);
    
    check_opengl_errors("upload texture!");


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
    
    GLenum internal_format = gl_internal_format(image->channels, image->type);

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

    GLenum format = gl_format(image->channels, image->type);
    GLenum datatype = gl_datatype(image->type);
    
    glReadPixels(0, 0, w, h, format, datatype, image->data);

    check_opengl_errors("read_pixels");

}

//////////////////////////////////////////////////////////////////////

GLuint make_shader(GLenum type,
                   GLint count,
                   const char** srcs,
                   const GLint* lengths) {

    require(lengths);

    GLuint shader = glCreateShader(type);
    glShaderSource(shader, count, srcs, lengths);
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

GLuint make_shader_sources(GLenum type,
                           GLint count,
                           const shader_source_t* sources) {

    buffer_t bufs[count];
    const char* srcs[count];
    GLint lengths[count];

    memset(bufs, 0, sizeof(bufs));

    for (GLint i=0; i<count; ++i) {
        require( sources[i].type == SHADER_SOURCE_STRING ||
                 sources[i].type == SHADER_SOURCE_FILE );
        if (sources[i].type == SHADER_SOURCE_STRING) {
            srcs[i] = sources[i].data;
            lengths[i] = sources[i].length ? sources[i].length : strlen(srcs[i]);
        } else {
            buf_append_file(bufs+i, sources[i].data, sources[i].length,
                            BUF_NULL_TERMINATE);
            srcs[i] = bufs[i].data;
            lengths[i] = bufs[i].size;
        }
    }
    
    GLuint shader = make_shader(type, count, srcs, lengths);

    for (GLint i=0; i<count; ++i) {
        buf_free(bufs+i);
    }

    return shader;
    
}
    
