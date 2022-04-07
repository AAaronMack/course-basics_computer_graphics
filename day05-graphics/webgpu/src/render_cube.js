import 'regenerator-runtime/runtime';

import { mat4, vec3 } from 'gl-matrix';

import {cubeVertexArray, cubeVertexSize, cubeUVOffset, cubePositionOffset, cubeVertexCount,} from './mesh_cube.js';

console.log("Log: ", cubeVertexCount);

var basicVertWGSL =
    `
        struct Uniforms {
          modelViewProjectionMatrix : mat4x4<f32>,
        };
        @binding(0) @group(0) var<uniform> uniforms : Uniforms;
        
        struct VertexOutput {
          @builtin(position) Position : vec4<f32>,
          @location(0) fragUV : vec2<f32>,
          @location(1) fragPosition: vec4<f32>,
        };
        
        @stage(vertex)
        fn main(@location(0) position : vec4<f32>,
                @location(1) uv : vec2<f32>) -> VertexOutput {
          var output : VertexOutput;
          output.Position = uniforms.modelViewProjectionMatrix * position;
          output.fragUV = uv;
          output.fragPosition = 0.5 * (position + vec4<f32>(1.0, 1.0, 1.0, 1.0));
          return output;
        }
    `;

var sampleTextureMixColorWGSL =
    `
        @group(0) @binding(1) var mySampler: sampler;
        @group(0) @binding(2) var myTexture: texture_2d<f32>;
        
        @stage(fragment)
        fn main(@location(0) fragUV: vec2<f32>,
                @location(1) fragPosition: vec4<f32>) -> @location(0) vec4<f32> {
          return textureSample(myTexture, mySampler, fragUV) * fragPosition;
        }
    `;

(async () => {
    var canvasRef = document.getElementById("webgpu-canvas-cube");
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    if (canvasRef === null) return;
    const context = canvasRef.getContext('webgpu');

    const devicePixelRatio = window.devicePixelRatio || 1;
    const presentationSize = [
        canvasRef.clientWidth * devicePixelRatio,
        canvasRef.clientHeight * devicePixelRatio,
    ];
    const presentationFormat = context.getPreferredFormat(adapter);

    context.configure({
        device,
        format: presentationFormat,
        size: presentationSize,
    });

    // Create a vertex buffer from the cube data.
    const verticesBuffer = device.createBuffer({
        size: cubeVertexArray.byteLength,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
    new Float32Array(verticesBuffer.getMappedRange()).set(cubeVertexArray);
    verticesBuffer.unmap();

    const pipeline = device.createRenderPipeline({
        vertex: {
            module: device.createShaderModule({
                code: basicVertWGSL,
            }),
            entryPoint: 'main',
            buffers: [
                {
                    arrayStride: cubeVertexSize,
                    attributes: [
                        {
                            // position
                            shaderLocation: 0,
                            offset: cubePositionOffset,
                            format: 'float32x4',
                        },
                        {
                            // uv
                            shaderLocation: 1,
                            offset: cubeUVOffset,
                            format: 'float32x2',
                        },
                    ],
                },
            ],
        },
        fragment: {
            module: device.createShaderModule({
                code: sampleTextureMixColorWGSL,
            }),
            entryPoint: 'main',
            targets: [
                {
                    format: presentationFormat,
                },
            ],
        },
        primitive: {
            topology: 'triangle-list',

            // Backface culling since the cube is solid piece of geometry.
            // Faces pointing away from the camera will be occluded by faces
            // pointing toward the camera.
            cullMode: 'back',
        },

        // Enable depth testing so that the fragment closest to the camera
        // is rendered in front.
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: 'depth24plus',
        },
    });

    const depthTexture = device.createTexture({
        size: presentationSize,
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    const uniformBufferSize = 4 * 16; // 4x4 matrix
    const uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Fetch the image and upload it into a GPUTexture.
    let cubeTexture;
    {
        const img = document.createElement('img');
        img.src = require('../img/girl.png');
        await img.decode();
        const imageBitmap = await createImageBitmap(img);

        cubeTexture = device.createTexture({
            size: [imageBitmap.width, imageBitmap.height, 1],
            format: 'rgba8unorm',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });
        device.queue.copyExternalImageToTexture(
            { source: imageBitmap },
            { texture: cubeTexture },
            [imageBitmap.width, imageBitmap.height]
        );
    }

    // Create a sampler with linear filtering for smooth interpolation.
    const sampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
    });

    const uniformBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: uniformBuffer,
                },
            },
            {
                binding: 1,
                resource: sampler,
            },
            {
                binding: 2,
                resource: cubeTexture.createView(),
            },
        ],
    });

    const renderPassDescriptor = {
        colorAttachments: [
            {
                view: undefined, // Assigned later

                clearValue: { r: 0.3, g: 0.3, b: 0.3, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            },
        ],
        depthStencilAttachment: {
            view: depthTexture.createView(),

            depthClearValue: 1.0,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
        },
    };

    const aspect = presentationSize[0] / presentationSize[1];
    const projectionMatrix = mat4.create();
    mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, aspect, 0.01, 100.0);

    function getTransformationMatrix() {
        const viewMatrix = mat4.create();
        mat4.translate(viewMatrix, viewMatrix, vec3.fromValues(0, 0, -4));
        const now = Date.now() / 1000;
        mat4.rotate(
            viewMatrix,
            viewMatrix,
            1,
            //vec3.fromValues(Math.sin(now), Math.cos(now), 0)
            vec3.fromValues(1,1,0)
        );
        mat4.scale(viewMatrix, viewMatrix, vec3.fromValues(1.25,1.25,1.25));

        const modelViewProjectionMatrix = mat4.create();
        mat4.multiply(modelViewProjectionMatrix, projectionMatrix, viewMatrix);

        return modelViewProjectionMatrix;
    }

    function frame() {
        // Sample is no longer the active page.
        if (!canvasRef) return;

        const transformationMatrix = getTransformationMatrix();
        device.queue.writeBuffer(
            uniformBuffer,
            0,
            transformationMatrix.buffer,
            transformationMatrix.byteOffset,
            transformationMatrix.byteLength
        );
        renderPassDescriptor.colorAttachments[0].view = context
            .getCurrentTexture()
            .createView();

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, uniformBindGroup);
        passEncoder.setVertexBuffer(0, verticesBuffer);
        passEncoder.draw(cubeVertexCount, 1, 0, 0);
        passEncoder.end();
        device.queue.submit([commandEncoder.finish()]);

        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
})();