#version 460

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 frag_uv;

layout(set = 1, binding = 0) uniform DrawData {
    mat4 vp;
} draw_data;

void main() {
    gl_Position = draw_data.vp * vec4(pos, 1.0); 
    // gl_Position = vec4(pos, 1.0) * draw_data.vp; 
    frag_uv = uv;
}
