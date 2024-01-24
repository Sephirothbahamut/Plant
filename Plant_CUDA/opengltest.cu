#include "OpenGLTest.cuh"

__global__ void createGPUTexture(uchar4* d_texture)
    {
    unsigned int pixelID = blockIdx.x * blockDim.x + threadIdx.x;
    d_texture[pixelID].x = 0;
    d_texture[pixelID].y = 1;
    d_texture[pixelID].z = 1;
    d_texture[pixelID].w = 1;
    }
__global__ void wow(uchar4* pos, unsigned int width, unsigned int height,
    float time)
    {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = index % width;
    unsigned int y = index / width;

    if (index < width * height) {
        unsigned char r = (x + (int)time) & 0xff;
        unsigned char g = (y + (int)time) & 0xff;
        unsigned char b = time;

        // Each thread writes one pixel location in the texture (textel)
        pos[index].w = 255;
        pos[index].x = r;
        pos[index].y = g;
        pos[index].z = b;
        }
    }
void OpenGLTest::drawFrame()
    {
    glColor3f(1.0f, 1.0f, 1.0f);

    glBindTexture(GL_TEXTURE_2D, gltexture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, windowSize.x, windowSize.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0.0f, float(windowSize.y));
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(float(windowSize.x), float(windowSize.y));
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(float(windowSize.x), 0.0f);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0.0f, 0.0f);
    glEnd();

    glFlush();

    // Release
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Test Triangle
    /*
    glBegin(GL_TRIANGLES);
    glColor3f(0.1, 0.2, 0.3);
    glVertex2f(0, 0);
    glVertex2f(10, 0);
    glVertex2f(0, 100);
    glEnd();
    */
    }

void OpenGLTest::createFrame(float time)
    {
    cudaGraphicsMapResources(1, &cudaPBO, 0);
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufferData, &numBytes, cudaPBO);

    int totalThreads = windowSize.x * windowSize.y;
    int nBlocks = totalThreads / 256;

    // Run code here.
    //createGPUTexture<<<nBlocks, 256>>> (d_textureBufferData);
    wow<<<nBlocks, 256>>>(d_textureBufferData, windowSize.x, windowSize.y, time);
    // Unmap mapping to PBO so that OpenGL can access.
    cudaGraphicsUnmapResources(1, &cudaPBO, 0);
    }

void OpenGLTest::setupOpenGL()
    {
    //image = new uchar4[1024 * 1024];
    image = new uchar4[windowSize.x * windowSize.y];
    glViewport(0, 0, windowSize.x, windowSize.y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, windowSize.x, windowSize.y, 0.0, -1.0, 1.0);

    glEnable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    // Unbind any textures from previous.
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // Create new textures.
    glGenTextures(1, &gltexture);
    glBindTexture(GL_TEXTURE_2D, gltexture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Create image with same resolution as window.
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowSize.x, windowSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);


    // Create pixel buffer boject.
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, windowSize.x * windowSize.y * sizeof(uchar4), image, GL_STREAM_COPY);

    cudaGraphicsGLRegisterBuffer(&cudaPBO, pbo, cudaGraphicsMapFlagsNone);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    }