#ifndef _FRAMEBUFFER_H_
#define _FRAMEBUFFER_H_

#include "gl_util.h"

typedef void (*glUniformFloatFunc)(GLint, GLsizei, const GLfloat*);
typedef void (*glUniformIntFunc)(GLint, GLsizei, const GLint*);

typedef enum uupdate_schedule {
    UUPDATE_ON_DRAW,
    UUPDATE_NOW,
} uupdate_schedule_t; 

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

enum {
    FB_MAX_INPUTS = 4,
    FB_MAX_UUPDATES = 4,
};

typedef struct framebuffer {

    const char* name;

    size_t width, height;
    GLenum internal_format;

    GLuint render_texture;
    GLuint framebuffer;

    size_t num_inputs;
    GLenum inputs[FB_MAX_INPUTS];

    GLuint program;

    size_t num_uupdates;
    uniform_update_t uupdates[FB_MAX_UUPDATES];

    int request_screenshot;

} framebuffer_t;

void fb_setup(framebuffer_t* fb,
              const char* name,
              size_t width, size_t height,
              GLenum internal_format,
              GLuint vertex_shader,
              GLuint fragment_shader);

void fb_add_input(framebuffer_t* fb,
                  const char* uniform_name,
                  GLuint texname);

void fb_screenshot(const framebuffer_t* fb);

void fb_uupdate_i(framebuffer_t* fb,
                  uupdate_schedule_t when,
                  const char* name, 
                  int array_length,
                  const int* src_int,
                  glUniformIntFunc int_func);

void fb_uupdate_f(framebuffer_t* fb,
                  uupdate_schedule_t when,
                  const char* name, 
                  int array_length,
                  const float* src_float,
                  glUniformFloatFunc float_func);

void fb_uupdate(framebuffer_t* fb,
                uupdate_schedule_t when,
                const uniform_update_t* uu);

void fb_uupdate_run(framebuffer_t* fb,
                    const uniform_update_t* uu);

void fb_run_uupdates(framebuffer_t* fb);

void fb_draw(framebuffer_t* fb);

#endif
