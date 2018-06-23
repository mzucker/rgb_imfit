out vec4 fragColor;

uniform sampler2D inputTexture;
uniform ivec2 inputDims;
uniform ivec2 outputDims;

void main() {

    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 tile_idx = p / outputDims;
    ivec2 p_rel = p - tile_idx * outputDims;

    ivec2 q0_rel = p_rel * SUM_WINDOW_SIZE;
    ivec2 q0_base = tile_idx * inputDims;

    fragColor = vec4(0);

    const float scl = 1. / float(SUM_WINDOW_SIZE * SUM_WINDOW_SIZE);

    // loop over window size
    for (int y=0; y<SUM_WINDOW_SIZE; ++y) {
        for (int x=0; x<SUM_WINDOW_SIZE; ++x) {

            ivec2 q = q0_rel + ivec2(x, y);

            if (q.x < inputDims.x && q.y < inputDims.y) {

                vec4 t = texelFetch(inputTexture, q + q0_base, 0);

                fragColor.xyz += t.xyz * t.w;
                fragColor.w += t.w;
                
            }

        }
    }

    
}
