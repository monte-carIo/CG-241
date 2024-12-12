#version 330 core

out vec4 fragColor;

void main() {
    float depth = ((gl_FragCoord.z/gl_FragCoord.w)/255);
    depth = 1 - depth;
    depth = (depth - 0.7)*5;
    fragColor = vec4(vec3(depth), 1.0);
}