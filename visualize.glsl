out vec4 fragColor;

uniform sampler2D srcTexture;
uniform sampler2D approxTexture;
uniform sampler2D errorTexture;

uniform ivec2 outputDims;

void main() {

    ivec2 p = ivec2(gl_FragCoord.xy);

    const ivec2 tiles = ivec2(3, 1);
    
    ivec2 srcDims = textureSize(srcTexture, 0);
    ivec2 tiledSrcDims = srcDims*tiles;

    ivec2 r = outputDims / srcDims;
    int repeat = max(min(r.x, r.y), 1);

    ivec2 m = (outputDims - repeat * tiledSrcDims) / 2;
    m = max(m, ivec2(0));

    p -= m;
    p /= repeat;

    if (p.x >= 0 && p.x < tiledSrcDims.x &&
        p.y >= 0 && p.y < tiledSrcDims.y) {

        int tile = p.x / srcDims.x;
        p.x -= tile * srcDims.x;

        p.y = tiledSrcDims.y - p.y - 1;

        if (tile == 0) {
            fragColor = texelFetch(srcTexture, p, 0);
        } else if (tile == 1) {
            fragColor = texelFetch(approxTexture, p, 0);
        } else {
            fragColor = texelFetch(errorTexture, p, 0);
            fragColor.xyz = sqrt(fragColor.xyz);
        }

    } else {
    
        fragColor = vec4(0);

    }

}
