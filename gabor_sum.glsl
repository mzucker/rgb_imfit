out vec4 fragColor;

uniform sampler2D evalTexture;
uniform vec2 srcDims;

void main() {

    ivec2 p0 = ivec2(gl_FragCoord.xy), p = p0;
    int sy = int(srcDims.y);

    vec3 accum = vec3(0);

    for (int i=0; i<EVAL_TILES; ++i) {

        accum += texelFetch(evalTexture, p, 0).xyz - 0.5;
        p.y += sy;
        
    }
    
    fragColor = vec4(accum + 0.5, 1);

    
}
