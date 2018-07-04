#include "framebuffer.h"
#include "require.h"

#include <string.h>
#include <stdio.h>

void fb_setup(framebuffer_t* fb,
              const char* name,
              size_t width, size_t height,
              GLenum internal_format,
              GLuint vertex_shader,
              GLuint fragment_shader) {

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
    
    check_opengl_errors("make program");

    fb->program = program;
    
}

//////////////////////////////////////////////////////////////////////

void fb_add_input(framebuffer_t* fb,
                  const char* uniform_name,
                  GLuint texname) {

    require(fb->num_inputs < FB_MAX_INPUTS);

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

void fb_uupdate_i(framebuffer_t* fb,
                  uupdate_schedule_t when,
                  const char* name, 
                  int array_length,
                  const int* src_int,
                  glUniformIntFunc int_func) {

    uniform_update_t uu;

    uu.type = GL_INT;
    uu.name = name;
    uu.array_length = array_length;
    uu.src_int = src_int;
    uu.int_func = int_func;

    fb_uupdate(fb, when, &uu);

}

//////////////////////////////////////////////////////////////////////

void fb_uupdate_f(framebuffer_t* fb,
                  uupdate_schedule_t when,
                  const char* name, 
                  int array_length,
                  const float* src_float,
                  glUniformFloatFunc float_func) {
    
    uniform_update_t uu;

    uu.type = GL_FLOAT;
    uu.name = name;
    uu.array_length = array_length;
    uu.src_float = src_float;
    uu.float_func = float_func;

    fb_uupdate(fb, when, &uu);

}

void fb_uupdate(framebuffer_t* fb,
                uupdate_schedule_t when,
                const uniform_update_t* uu) {

    require(when == UUPDATE_NOW || when == UUPDATE_ON_DRAW);

    if (when == UUPDATE_NOW) {
        
        fb_uupdate_run(fb, uu);
        
    } else {

        require( fb->num_uupdates < FB_MAX_UUPDATES );
        fb->uupdates[fb->num_uupdates++] = *uu;

    }
    
}

//////////////////////////////////////////////////////////////////////

void fb_run_uupdates(framebuffer_t* fb) {
    
    for (size_t i=0; i<fb->num_uupdates; ++i) {

        const uniform_update_t* uu = fb->uupdates + i;
        fb_uupdate_run(fb, uu);

    }

    memset(fb->uupdates, 0, sizeof(fb->uupdates));
    fb->num_uupdates = 0;

}

//////////////////////////////////////////////////////////////////////

void fb_uupdate_run(framebuffer_t* fb,
                    const uniform_update_t* uu) {
    
    require(uu->type == GL_INT || uu->type == GL_FLOAT);
    
    int location = glGetUniformLocation(fb->program, uu->name);
    if (location == -1) { return; }
    
    if (uu->type == GL_INT) {
        uu->int_func(location, uu->array_length, uu->src_int);
    } else {
        uu->float_func(location, uu->array_length, uu->src_float);
    }

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
