#ifndef _GL_UTIL_H_
#define _GL_UTIL_H_

#include <GL/glew.h>
#include "image.h"

typedef enum shader_source_type {
    SHADER_SOURCE_STRING,
    SHADER_SOURCE_FILE,
} shader_source_type_t;

typedef struct shader_source {
    shader_source_type_t type;
    const char* data;
    size_t length;
} shader_source_t;

const char* get_error_string(GLenum error);
void check_opengl_errors(const char* context);

GLenum gl_datatype(image_type_t type);
GLenum gl_format(size_t channels, image_type_t datatype);
GLenum gl_internal_format(size_t channels, image_type_t datatype);

void upload_texture(const image_t* image);
GLuint make_texture(image_t* image);

void read_pixels(image_t* image);

GLuint make_shader(GLenum type,
                   GLint count,
                   const char** srcs,
                   const GLint* lengths);

GLuint make_shader_sources(GLenum type,
                           GLint count,
                           const shader_source_t* sources);

#endif
