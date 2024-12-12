#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 texCoord;

uniform mat4 projection;
uniform mat4 modelview;

out vec2 TexCoord;
out float max;
out float min;

void main() {
    gl_Position = projection * modelview * vec4(position, 1.0);
    TexCoord = texCoord;
}