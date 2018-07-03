out vec4 fragColor;

uniform usampler2D paramTexture;
uniform vec2 srcDims;

bool assert_fail = false;

#define assert(x) if (!(x)) { assert_fail = true; }

#define TWOPI 6.28318530718

vec3 unpack(uint x) {
    
    uvec3 u = (uvec3(x) >> uvec3(0, 10, 20)) & uint(1023);
    vec3 v = vec3(u);

    assert(v.x >= 0 && v.x <= 1023);
    assert(v.y >= 0 && v.y <= 1023);
    assert(v.z >= 0 && v.z <= 1023);
    
    return v;
    
}

void main() {
    
    float maxDim = max(srcDims.x, srcDims.y);
    
    float px = 2.0/maxDim;

    vec2 f = mod(gl_FragCoord.xy, srcDims);

    int param_idx = int(floor( gl_FragCoord.y / srcDims.y ) );
    
    vec2 p = (f - 0.5*srcDims)*px;

    int j = int(floor(f.y)*srcDims.x + floor(f.x));
    
    uvec4 data = texelFetch(paramTexture, ivec2(j, param_idx), 0);

    //fragColor = vec4(vec3(data.xyz)/vec3(uvec3(-1)), 1);

    vec3 accum = vec3(0);

    for (int j=0; j<GABORS_PER_TILE; ++j) {

        uvec4 params = texelFetch(paramTexture, ivec2(j, param_idx), 0);

        vec2 uv = mix(vec2(-1.5), vec2(1.5), unpack(params[0]).xy/1023.);
        vec2 phir = unpack(params[1]).xy * (TWOPI / 1024.);
        vec3 h = unpack(params[2])/1023;
        h *= 0.5 * h; // bias h smaller!
        
        vec3 stl_raw = unpack(params[3])/1023;
        
        stl_raw.x *= stl_raw.x; // bias s smaller!

        float s = mix(px, 2., stl_raw.x);
        
        
        float t = mix(s, s*4., stl_raw.y);
        float l = mix(s*2., s*8., stl_raw.z);
        
        float s2 = s*s;
        float t2 = t*t;

        float cr = cos(phir.y);
        float sr = sin(phir.y);
        
        mat2 R = mat2(cr, sr, -sr, cr);

        vec2 q = R * (p - uv);

        float f = 6.283185307179586 / l;

        float ck = cos(f*q.x + phir.x);
        float w = exp(-q.x*q.x/(2.*s2) - q.y*q.y/(2.*t2));

        vec3 g = h * w * ck;
        
        accum += 0.5*g;

        
    }

    fragColor = vec4(accum + 0.5, 1);

    if (assert_fail) { fragColor = vec4(1, 0, 0, 1); }
    
}
