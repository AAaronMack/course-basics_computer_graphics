// FIX regeneratorRuntime is not defined error
//  https://dev.to/hulyakarakaya/how-to-fix-regeneratorruntime-is-not-defined-doj
// import "babel-polyfill";
import 'regenerator-runtime/runtime';

// Test wgpu-matrix module
import { mat4 } from "wgpu-matrix";

const fov = 60 * Math.PI / 180
const aspect = 1920 / 1080;
const near = 0.1;
const far = 1000;
const perspective = mat4.perspective(fov, aspect, near, far);
const mat4i = mat4.identity();
console.log("Log: ", perspective);
console.log("Log: ", mat4i);


(async () => {
    // 1.Determine whether the browser supports WebGPU
    if (navigator.gpu === undefined) {
        document.getElementById("webgpu-canvas-triangle").setAttribute("style", "display:none;");
        document.getElementById("no-webgpu").setAttribute("style", "display:block;");
        return;
    }

    // 2.Get a GPU device to render with
    //  An Adapter describes the physical properties of a given GPU, such as its name, extensions, and device limits.
    var adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        document.getElementById("webgpu-canvas-triangle").setAttribute("style", "display:none;");
        document.getElementById("no-webgpu").setAttribute("style", "display:block;");
        return;
    }
    // A Device is how you access the core of the WebGPU API, and will allow you to create the data structures you'll need.
    var device = await adapter.requestDevice();

    // 3.Get a context to display our rendered image on the canvas
    // In order to see what you're drawing, you'll need an HTMLCanvasElement and to setup a Canvas Context from that canvas.
    // A Canvas Context manages a series of textures you'll use to present your final render output to your <canvas> element.
    var canvas = document.getElementById("webgpu-canvas-triangle");
    var context = canvas.getContext("webgpu");

    var swapChainFormat = "bgra8unorm";
    var depthFormat = "depth24plus";

    const canvasConfig = {
        device: device,
        format: swapChainFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
    };
    context.configure(canvasConfig);

    // A Queue allows you to send work asynchronously to the GPU.
    var queue = device.queue;


    // 4.Setup our shader modules
    var vertShaderCode =
        `
    type float4 = vec4<f32>;
    struct VertexInput {
        @location(0) position: float4,
        @location(1) color: float4,
    };

    struct VertexOutput {
        @builtin(position) position: float4,
        @location(0) color: float4,
    };

    @stage(vertex)
    fn vertex_main(vert: VertexInput) -> VertexOutput {
        var out: VertexOutput;
        out.color = vert.color;
        out.position = vert.position;
        return out;
    };
    `;

    var fragShaderCode =
        `
    type float4 = vec4<f32>;
    struct VertexOutput {
        @builtin(position) position: float4,
        @location(0) color: float4,
    };

    @stage(fragment)
    fn fragment_main(in: VertexOutput) -> @location(0) float4 {
        return float4(in.color);
    }
    `;

    var vertModule = device.createShaderModule({code: vertShaderCode});
    var fragModule = device.createShaderModule({code: fragShaderCode});

    const _shaderModules = {
        "vert": vertModule,
        "frag": fragModule
    };

    // 4.1 Check whether there are errors when compiling shader
    for(const [key, value] of Object.entries(_shaderModules)){
        if (value.compilationInfo) {
            var compilationInfo = await value.compilationInfo();
            if (compilationInfo.messages.length > 0) {
                var hadError = false;
                console.log("Shader compilation log: ", key);

                for (var i = 0; i < compilationInfo.messages.length; ++i) {
                    var msg = compilationInfo.messages[i];
                    console.log(`${msg.lineNum}:${msg.linePos} - ${msg.message}`);
                    hadError = hadError || msg.type == "error";
                }
                if (hadError) {
                    console.error("Shader failed to compile");
                    return;
                }
            }
        }
    }

    // 5.Specify vertex data
    // 5.1 Allocating empty space will fill data later
    var dataBuf = device.createBuffer({
        size: 3 * 2 * 4 * 4,  // for (3 * position + 3 * color) * four elements * 4 bytes
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });

    // 5.2 Filling data (Feed the data into the machine.)
    // The `position` and `color` orders are same as `struct VertexInput`
    new Float32Array(dataBuf.getMappedRange()).set([
        1.0, -1, 1, 1,  // position
        1, 0, 0, 1,   // color
        -1, -1, 0, 1, // position
        0, 1, 0, 1,   // color
        0, 1, 0, 1,   // position
        0, 0, 1, 1,   // color
    ]);
    // Tell the GPU we're done filling
    dataBuf.unmap();

    // 5.3 Setup data (Like some description of the connection between our data and shaders)
    // Vertex attribute state and shader stage
    var vertexState = {
        module: vertModule,
        entryPoint: "vertex_main",
        buffers: [{
            arrayStride: 2 * 4 * 4,
            attributes: [
                {format: "float32x4", offset: 0, shaderLocation: 0},
                {format: "float32x4", offset: 4 * 4, shaderLocation: 1}
            ]
        }]
    };
    // Fragment output targets and shader stage
    var fragmentState = {
        module: fragModule,
        entryPoint: "fragment_main",
        targets: [{format: swapChainFormat}]
    };

    // 6.Create render pipeline
    var renderPipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({bindGroupLayouts: []}),
        vertex: vertexState,
        fragment: fragmentState,
        depthStencil: {format: depthFormat, depthWriteEnabled: true, depthCompare: "less"}
    });

    var depthTexture = device.createTexture({
        size: {width: canvas.width, height: canvas.height, depth: 1},
        format: depthFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });

    var renderPassDesc = {
        colorAttachments: [{
            view: undefined,
            clearValue: { r: 0.3, g: 0.3, b: 0.3, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
        }],   // Background canvas
        depthStencilAttachment: {       // Depth canvas (It's like the order when you draw a picture.)
            view: depthTexture.createView(),
            depthClearValue: 1.0,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
        }
    };

    // Track when the canvas is visible
    // on screen, and only render when it is visible.
    var canvasVisible = false;
    var observer = new IntersectionObserver(function(e) {
        if (e[0].isIntersecting) {
            canvasVisible = true;
        } else {
            canvasVisible = false;
        }
    }, {threshold: [0]});
    observer.observe(canvas);

    var frame = function() {
        if (canvasVisible) {
            renderPassDesc.colorAttachments[0].view = context.getCurrentTexture().createView();

            var commandEncoder = device.createCommandEncoder();

            var renderPass = commandEncoder.beginRenderPass(renderPassDesc);

            // Draw the body of the picture.
            renderPass.setPipeline(renderPipeline);
            renderPass.setVertexBuffer(0, dataBuf);
            renderPass.draw(3, 1, 0, 0);

            renderPass.end();
            queue.submit([commandEncoder.finish()]);
        }
        requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
})();
