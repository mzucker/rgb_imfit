out vec4 fragColor;

uniform sampler2D paramTexture;
uniform vec2 srcDims;

void main() {
    
    float maxDim = max(srcDims.x, srcDims.y);
    
    float px = 2.0/maxDim;

    vec2 f = mod(gl_FragCoord.xy, srcDims);

    int param_idx = int(floor( gl_FragCoord.x / srcDims.x ) );
    
    vec2 p = (f - 0.5*srcDims)*px;

    vec3 accum = vec3(0);

    int k = 3*param_idx;

    for (int j=0; j<NUM_PARAMS; ++j) {
        
        vec4 uvst = texelFetch(paramTexture, ivec2(k+0, j), 0);
        vec4 phir = texelFetch(paramTexture, ivec2(k+1, j), 0);
        vec4 hl   = texelFetch(paramTexture, ivec2(k+2, j), 0);

        float s = clamp(uvst.z, px, 2.0);
        float t = clamp(uvst.w, px, 4.0);
        float l = clamp(hl.w, 2.5*px, 4.0);
        vec3 h = clamp(hl.xyz, 0.0, 2.0);

        float s2 = s*s;
        float t2 = t*t;

        float cr = cos(phir.w);
        float sr = sin(phir.w);
        
        mat2 R = mat2(cr, sr, -sr, cr);

        vec2 q = R * (p - uvst.xy);

        float f = 6.283185307179586 / l;

        vec3 ck = cos(f*q.x + phir.xyz);
        float w = exp(-q.x*q.x/(2.*s2) - q.y*q.y/(2.*t2));

        vec3 g = h * w * ck;

        accum += 0.5*g;
        
    }

    fragColor = vec4(accum + 0.5, 1);
    
}
