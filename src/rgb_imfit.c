#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "require.h"
#include "buffer.h"
#include "image.h"
#include "gl_util.h"
#include "framebuffer.h"
#include "pcg_tiny.h"

/*  TODO: 

    - load/save params
    - more efficient bit encoding (have 128 bits, using just 100)
    - implement command line switches
    - use YUV conversion from https://en.wikipedia.org/wiki/YUV#HDTV_with_BT.709
    - use weights on YUV from http://richg42.blogspot.com/2018/04/bc7-encoding-using-weighted-ycbcr.html
    - ESC to quit?
    - GA instead of annealing?
    - figure out how to better capture fine detail?
    - resize input using mipmapping

   DONE:

    - use ints instead of floats (~10 bits/param)
    - split into multiple source files

*/

//////////////////////////////////////////////////////////////////////
// TYPEDEFS/ENUMS

enum {
    VIS_TILES = 2,
    MAX_SOURCE_LENGTH = 1024*1024,
    SUM_WINDOW_SIZE = 2
};

typedef enum solver_type {
    SOLVER_ANNEALING,
    SOLVER_GA,
} solver_type_t;

size_t gabors_per_tile = 128;

#ifdef __APPLE__
size_t num_tiles = 100;
size_t num_profile = 10;
#else
size_t num_tiles = 100;
size_t num_profile = 1000;
#endif

solver_type_t solver;

GLuint vertex_shader;

char common_defines[1024];
size_t common_defines_length;

image_t src_image32f;
image_t weight_image32f;
image_t palette_image32f;

image_t param_image32u;

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
    GABOR_PARAM_RESERVED0,
    
    GABOR_PARAM_PHI,
    GABOR_PARAM_R,
    GABOR_PARAM_RESERVED1,
    
    GABOR_PARAM_H0,
    GABOR_PARAM_H1,
    GABOR_PARAM_H2,
    
    GABOR_PARAM_S,
    GABOR_PARAM_T,
    GABOR_PARAM_L,

    GABOR_NUM_PARAMS, 
    
    GABOR_PARAM_BITS = 10,
    GABOR_PARAMS_PER_UINT32 = 3,
    GABOR_UINT32_ELEMENTS = 4,
    GABOR_PARAM_MASK = (1 << GABOR_PARAM_BITS) - 1,
    
};

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

void error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW error: %s\n", description);
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

    fb_uupdate_i(&main_fb, UUPDATE_ON_DRAW,
                 "outputDims", 1, odims, glUniform2iv);
    
    fb_draw(&main_fb);

    if (window) {
        
        main_fb.framebuffer = old_framebuffer;
        main_fb.width = old_width;
        main_fb.height = old_height;

    }

    
}

//////////////////////////////////////////////////////////////////////

void init_params(uint32_t pi[GABOR_UINT32_ELEMENTS]) {

    for (int j=0; j<GABOR_UINT32_ELEMENTS; ++j) {
        pi[j] = pcg32_random();
    }
    
}


//////////////////////////////////////////////////////////////////////

void mutate_params(uint32_t pi[GABOR_UINT32_ELEMENTS], float amount) {

    int pidx = 0;

    for (int el=0; el<GABOR_UINT32_ELEMENTS; ++el) {
        
        uint32_t u_in = pi[el];
        uint32_t u_out = 0;
        
        for (int j=0; j<GABOR_PARAMS_PER_UINT32; ++j) {
            
            uint32_t param = (u_in >> j*GABOR_PARAM_BITS) & GABOR_PARAM_MASK;

            int perturb = (signed_random3() * amount) * 1023;

            if (pidx == GABOR_PARAM_PHI || pidx == GABOR_PARAM_R) {
                
                param = (param + perturb) & GABOR_PARAM_MASK;
                
            } else if (pidx != GABOR_PARAM_RESERVED0 &&
                       pidx != GABOR_PARAM_RESERVED1) {
                
                int32_t sparam = (int32_t)param + perturb;
                
                sparam = (sparam < 0 ? 0 :
                          sparam > GABOR_PARAM_MASK ? GABOR_PARAM_MASK :
                          sparam);
                
                param = sparam;
                
            }
        
            require( param <= GABOR_PARAM_MASK );

            u_out |= (param << j*GABOR_PARAM_BITS);

            ++pidx;
            
        }

        pi[el] = u_out;
        
    }

        require(pidx == GABOR_NUM_PARAMS);
    
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

    require( GABOR_UINT32_ELEMENTS == 4 );
    require( GABOR_PARAM_BITS * GABOR_PARAMS_PER_UINT32 < 32 );
    require( GABOR_PARAM_BITS * GABOR_NUM_PARAMS < 32 * GABOR_UINT32_ELEMENTS );
             
    image_create(&param_image32u, gabors_per_tile, num_tiles,
                 GABOR_UINT32_ELEMENTS, IMAGE_32U);

    for (int pidx=0; pidx<num_tiles; ++pidx) {
        for (int midx=0; midx<gabors_per_tile; ++midx) {
            int i = pidx * gabors_per_tile + midx;
            uint32_t* pi = param_image32u.data_32u + i*GABOR_UINT32_ELEMENTS;
            init_params(pi);
        }
    }

    return make_texture(&param_image32u);

}

//////////////////////////////////////////////////////////////////////

void setup_vertex_stuff() {

    shader_source_t source = {
        SHADER_SOURCE_FILE, "../shaders/vertex.glsl", 1024
    };

    vertex_shader = make_shader_sources(GL_VERTEX_SHADER, 1, &source);

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

void setup_fb_simple(framebuffer_t* fb, const char* name,
                     size_t w, size_t h,
                     const char* filename) {

    
    shader_source_t sources[2] = {
        { SHADER_SOURCE_STRING, common_defines, common_defines_length },
        { SHADER_SOURCE_FILE, filename, MAX_SOURCE_LENGTH }
    };

    GLuint frament_shader = make_shader_sources(GL_FRAGMENT_SHADER, 2, sources);

    fb_setup(fb, name, w, h, GL_RGBA32F, vertex_shader, frament_shader);

    GLint vpos_location = glGetAttribLocation(fb->program, "vertexPosition");
    require(vpos_location != -1);
    
    glEnableVertexAttribArray(vpos_location);
    glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE,
                          sizeof(float) * 2, (void*) 0);

}

//////////////////////////////////////////////////////////////////////

void setup_framebuffers(GLFWwindow* window) {

    snprintf(common_defines, sizeof(common_defines),
             "#version 330\n"
             "#define GABORS_PER_TILE %d\n"
             "#define SUM_WINDOW_SIZE %d\n",
             (int)gabors_per_tile, SUM_WINDOW_SIZE);

    printf("common_defines is:\n%s", common_defines);

    common_defines_length = strlen(common_defines);

    ////////////////////////////////////////////////////////////

    setup_fb_simple(&gabor_eval_fb, "gabor_eval",
                    src_image32f.width,
                    src_image32f.height * num_tiles,
                    "../shaders/gabor_eval.glsl");
    
    fb_add_input(&gabor_eval_fb, "paramTexture", param_image32u.bound_texture);

    float fdims[2] = { src_image32f.width, src_image32f.height };
    
    fb_uupdate_f(&gabor_eval_fb, UUPDATE_NOW,
                 "srcDims", 1, fdims, glUniform2fv);

    
    ////////////////////////////////////////////////////////////
    
    setup_fb_simple(&gabor_compare_fb, "gabor_compare",
                    src_image32f.width,
                    src_image32f.height * num_tiles,
                    "../shaders/gabor_compare.glsl");

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
        
        setup_fb_simple(reduce_fbs + i, reduce_name,
                        w, h * num_tiles,
                        "../shaders/reduce.glsl");

        fb_add_input(reduce_fbs + i,
                     "inputTexture",
                     input_texture);

        int idims[2] = { prev_w, prev_h };
        int odims[2] = { w, h };
        
        fb_uupdate_i(reduce_fbs + i, UUPDATE_NOW,
                     "inputDims", 1,
                     idims, glUniform2iv);

        fb_uupdate_i(reduce_fbs + i, UUPDATE_NOW,
                     "outputDims", 1,
                     odims, glUniform2iv);
        
        input_texture = reduce_fbs[i].render_texture;

    }

    image_create(&reduced_image32f, 1, num_tiles, 4, IMAGE_32F);
    printf("reduced_image32f has size %d\n", (int)reduced_image32f.buf.size);

    objective_values = malloc(num_tiles*sizeof(float));

    setup_fb_simple(&main_fb, "main",
                    VIS_TILES*src_image32f.width,
                    src_image32f.height,
                    "../shaders/visualize.glsl");

    fb_add_input(&main_fb, "srcTexture", src_image32f.bound_texture);
    fb_add_input(&main_fb, "approxTexture", gabor_eval_fb.render_texture);
    fb_add_input(&main_fb, "errorTexture", gabor_compare_fb.render_texture);
    fb_add_input(&main_fb, "palette", palette_image32f.bound_texture);

    int sdims[2] = { src_image32f.width, src_image32f.height };
    int ntiles[2] = { VIS_TILES, num_tiles };

    fb_uupdate_i(&main_fb, UUPDATE_NOW, "srcDims", 1, sdims, glUniform2iv);

    fb_uupdate_i(&main_fb, UUPDATE_NOW, "numTiles", 1, ntiles, glUniform2iv);

}

//////////////////////////////////////////////////////////////////////

void compute() {

    upload_texture(&param_image32u);

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

    anneal.t_max = 5e-6;
    
    const double max_iter = 1e9;
    const double temp_decay = 1e-3;
    anneal.t_rate = -log(temp_decay) / max_iter;

    anneal.change_fraction = 0.0;
    anneal.p_reinitialize = 0.01;
    anneal.mutate_amount = 0.01;

    image_copy(&anneal.good_params32f, &param_image32u);

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
               (const char*)param_image32u.buf.data,
               param_image32u.buf.size);
        
    } else {
        
        memcpy(param_image32u.buf.data,
               (const char*)anneal.good_params32f.buf.data,
               param_image32u.buf.size);
        
    }


    // tweak multiple gabor functions
    size_t nchange = floor(anneal.change_fraction * gabors_per_tile + 0.5);
    if (nchange < 1) { nchange = 1; }
        

    require( num_tiles == 1 );

    for (size_t pidx=0; pidx<num_tiles; ++pidx) {
        for (size_t c=0; c<nchange; ++c) {

            int midx = pcg32_random_r(&rng_global) % gabors_per_tile;
            int i = midx + pidx * gabors_per_tile;
        
            uint32_t* pi = param_image32u.data_32u + i*GABOR_UINT32_ELEMENTS;
        
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

    const size_t iter_per_vis = 5000;
    const int screenshots_enabled = 1;

    int num_screenshots = 0;

    while (1) {

        compute();
        
        int do_vis = (iteration % iter_per_vis == 0);

        if (do_vis) {
 
            double elapsed = glfwGetTime() - start;
            anneal_info(iteration, elapsed, iter_since_printout);
                   
            draw_main(window);
            glfwSwapBuffers(window);
            glfwPollEvents();
            if (glfwWindowShouldClose(window)) { break; }

            if (screenshots_enabled) {
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

