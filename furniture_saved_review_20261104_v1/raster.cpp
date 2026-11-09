// CPU depth-buffer renderer for recorded collision triangles. Screen x/y and
// depth are in pixel units; larger depth is closer to the viewer.
#include <algorithm>
#include <cmath>
#include <cstdint>

extern "C" void paint(uint8_t *rgb, float *depth, int width, int height,
                      const float *verts, const int32_t *faces,
                      const uint8_t *colors, int count) {
    for (int i = 0; i < count; ++i) {
        const float *a = verts + 3 * faces[3*i];
        const float *b = verts + 3 * faces[3*i+1];
        const float *c = verts + 3 * faces[3*i+2];
        const float dx = b[0]-a[0], dy = b[1]-a[1], dz = b[2]-a[2];
        const float ex = c[0]-a[0], ey = c[1]-a[1], ez = c[2]-a[2];
        const float det = dx*ey-dy*ex;
        if (std::abs(det) < 1e-6f) continue;
        const int left = std::max(0, (int)std::ceil(std::min({a[0],b[0],c[0]})-.5f));
        const int right = std::min(width-1, (int)std::floor(std::max({a[0],b[0],c[0]})-.5f));
        const int top = std::max(0, (int)std::ceil(std::min({a[1],b[1],c[1]})-.5f));
        const int bottom = std::min(height-1, (int)std::floor(std::max({a[1],b[1],c[1]})-.5f));
        if (left>right || top>bottom) continue;
        const float nx=dy*ez-dz*ey, ny=dz*ex-dx*ez;
        const float norm=std::sqrt(nx*nx+ny*ny+det*det);
        const float lambert=std::copysign(1.f,det)*(-.35f*nx-.62f*ny+.70f*det)/norm;
        const float shade=.58f+.42f*std::max(0.f,lambert);
        uint8_t pixel[3];
        for (int k=0;k<3;++k) pixel[k]=(uint8_t)(colors[3*i+k]*shade);
        for (int y=top;y<=bottom;++y) for (int x=left;x<=right;++x) {
            const float px=x+.5f-a[0], py=y+.5f-a[1];
            const float u=(px*ey-py*ex)/det, v=(dx*py-dy*px)/det;
            if (u<0 || v<0 || u+v>1) continue;
            const float z=a[2]+u*dz+v*ez;
            const int index=y*width+x;
            if (z<=depth[index]) continue;
            depth[index]=z;
            for (int k=0;k<3;++k) rgb[3*index+k]=pixel[k];
        }
    }
}
