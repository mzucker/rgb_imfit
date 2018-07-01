out vec4 fragColor;

uniform sampler2D paramTexture;
uniform vec2 srcDims;

void main() {
    
    float maxDim = max(srcDims.x, srcDims.y);
    
    float px = 2.0/maxDim;

    vec2 f = mod(gl_FragCoord.xy, srcDims);

    int param_idx = int(floor( gl_FragCoord.y / srcDims.y ) );
    
    vec2 p = (f - 0.5*srcDims)*px;

    vec3 accum = vec3(0);

    for (int j=0; j<GABORS_PER_TILE; ++j) {

        int pstart = 3*j;
        
        vec4 uvst = texelFetch(paramTexture, ivec2(pstart+0, param_idx), 0);
        vec4 phir = texelFetch(paramTexture, ivec2(pstart+1, param_idx), 0);
        vec4 hl   = texelFetch(paramTexture, ivec2(pstart+2, param_idx), 0);

        
        float s = clamp(uvst.z, px, 2.0);
        float t = clamp(uvst.w, px, 4.0);
        float l = clamp(hl.w, 2.5*px, 4.0);
        vec3 h = clamp(hl.xyz, 0.0, 2.0);
        //float h = clamp(hl.x, 0.0, 2.0);

        float s2 = s*s;
        float t2 = t*t;

        float cr = cos(phir.w);
        float sr = sin(phir.w);
        
        mat2 R = mat2(cr, sr, -sr, cr);

        vec2 q = R * (p - uvst.xy);

        float f = 6.283185307179586 / l;

        float ck = cos(f*q.x + phir.x);
        //vec3 ck = cos(f*q.x + phir.xyz);
        float w = exp(-q.x*q.x/(2.*s2) - q.y*q.y/(2.*t2));

        vec3 g = h * w * ck;
        
        accum += 0.5*g;
        
    }

    fragColor = vec4(accum + 0.5, 1);
    
}
