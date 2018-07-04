out vec4 fragColor;

uniform sampler2D approxTexture;
uniform sampler2D srcTexture;
uniform sampler2D weightTexture;

void main() {

    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 p0 = p % textureSize(srcTexture, 0);

    vec3 approx = clamp(texelFetch(approxTexture, p, 0).xyz, 0., 1.);
    vec3 src = clamp(texelFetch(srcTexture, p0, 0).xyz, 0., 1.);
    float w = clamp(texelFetch(weightTexture, p0, 0).x, 0., 1.);

    w *= w;
    
    vec3 diff = src - approx;

    fragColor = vec4(vec3(dot(diff, diff)*w), w);

}
