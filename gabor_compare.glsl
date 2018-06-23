out vec4 fragColor;

uniform sampler2D approxTexture;
uniform sampler2D srcTexture;

void main() {

    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 p0 = p % textureSize(srcTexture, 0);

    vec3 approx = clamp(texelFetch(approxTexture, p, 0).xyz, 0., 1.);
    vec3 src = clamp(texelFetch(srcTexture, p0, 0).xyz, 0., 1.);

    vec3 diff = src - approx;

    // TODO shove weight in alpha component
    fragColor = vec4(vec3(dot(diff, diff)), 1);

}
