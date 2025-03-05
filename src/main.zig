const std = @import("std");

const math = @import("math.zig");
const sdl = @import("sdl.zig");

const Input = @import("input.zig");

const log = std.log;

pub const ticks_per_second = 43;
pub const tick: f32 = 1.0 / @as(f32, @floatFromInt(ticks_per_second));
pub const tick_ns: u64 = 1000_000_000 / ticks_per_second;
pub const max_tick_ns: u64 = @intFromFloat(0.1 * 1e9);

pub fn main() !void {
    var gpa_struct: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_struct.deinit();
    const gpa = gpa_struct.allocator();

    sdl.c.SDL_SetMainReady();

    if (!sdl.c.SDL_Init(sdl.c.SDL_INIT_VIDEO)) {
        log.err("SDL_Init: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    }
    defer sdl.c.SDL_Quit();

    const window = sdl.c.SDL_CreateWindow("audio-raytracing", 800, 600, 0) orelse {
        log.err("SDL_CreateWindow: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    };
    defer sdl.c.SDL_DestroyWindow(window);

    const gpu_device = sdl.c.SDL_CreateGPUDevice(
        sdl.c.SDL_GPU_SHADERFORMAT_SPIRV,
        true,
        null,
    ) orelse {
        log.err("SDL_CreateGPUDevice: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    };
    defer sdl.c.SDL_DestroyGPUDevice(gpu_device);

    if (!sdl.c.SDL_ClaimWindowForGPUDevice(gpu_device, window)) {
        log.err("SDL_ClaimWindowForGPUDevice: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    }

    const vert_shader = blk: {
        const file = try std.fs.cwd().openFile(
            "data/shaders/shader.vert.spv",
            .{ .mode = .read_only },
        );
        defer file.close();
        const bytes = try file.reader().readAllAlloc(gpa, 1_000_000);
        defer gpa.free(bytes);

        const create_info = sdl.c.SDL_GPUShaderCreateInfo{
            .code_size = bytes.len,
            .code = bytes.ptr,
            .entrypoint = "main",
            .format = sdl.c.SDL_GPU_SHADERFORMAT_SPIRV,
            .stage = sdl.c.SDL_GPU_SHADERSTAGE_VERTEX,
            .num_samplers = 0,
            .num_storage_textures = 0,
            .num_storage_buffers = 0,
            .num_uniform_buffers = 0,
        };
        break :blk sdl.c.SDL_CreateGPUShader(gpu_device, &create_info) orelse {
            log.err("SDL_CreateGPUShader: {s}", .{sdl.c.SDL_GetError()});
            return error.Sdl;
        };
    };

    const frag_shader = blk: {
        const file = try std.fs.cwd().openFile(
            "data/shaders/shader.frag.spv",
            .{ .mode = .read_only },
        );
        defer file.close();
        const bytes = try file.reader().readAllAlloc(gpa, 1_000_000);
        defer gpa.free(bytes);

        const create_info = sdl.c.SDL_GPUShaderCreateInfo{
            .code_size = bytes.len,
            .code = bytes.ptr,
            .entrypoint = "main",
            .format = sdl.c.SDL_GPU_SHADERFORMAT_SPIRV,
            .stage = sdl.c.SDL_GPU_SHADERSTAGE_FRAGMENT,
            .num_samplers = 1,
            .num_storage_textures = 0,
            .num_storage_buffers = 0,
            .num_uniform_buffers = 1,
        };
        break :blk sdl.c.SDL_CreateGPUShader(gpu_device, &create_info) orelse {
            log.err("SDL_CreateGPUShader: {s}", .{sdl.c.SDL_GetError()});
            return error.Sdl;
        };
    };

    const vertex_buffer_descriptions = [_]sdl.c.SDL_GPUVertexBufferDescription{.{
        .slot = 0,
        .input_rate = sdl.c.SDL_GPU_VERTEXINPUTRATE_VERTEX,
        .instance_step_rate = 0,
        .pitch = @sizeOf(Vertex),
    }};
    const vertex_attributes = [_]sdl.c.SDL_GPUVertexAttribute{ .{
        .buffer_slot = 0,
        .format = sdl.c.SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
        .location = 0,
        .offset = @offsetOf(Vertex, "pos"),
    }, .{
        .buffer_slot = 0,
        .format = sdl.c.SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
        .location = 1,
        .offset = @offsetOf(Vertex, "uv"),
    } };

    const main_color_target_descriptions = [_]sdl.c.SDL_GPUColorTargetDescription{.{
        .format = sdl.c.SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .blend_state = .{},
    }};
    const main_gpu_pipeline_create_info = sdl.c.SDL_GPUGraphicsPipelineCreateInfo{
        .vertex_shader = vert_shader,
        .fragment_shader = frag_shader,
        .vertex_input_state = .{
            .vertex_buffer_descriptions = &vertex_buffer_descriptions[0],
            .num_vertex_buffers = vertex_buffer_descriptions.len,
            .vertex_attributes = &vertex_attributes[0],
            .num_vertex_attributes = vertex_attributes.len,
        },
        .primitive_type = sdl.c.SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .rasterizer_state = .{},
        .multisample_state = .{},
        .depth_stencil_state = .{
            .compare_op = sdl.c.SDL_GPU_COMPAREOP_LESS,
            .enable_depth_test = true,
            .enable_depth_write = true,
        },
        .target_info = .{
            .color_target_descriptions = &main_color_target_descriptions[0],
            .num_color_targets = main_color_target_descriptions.len,
            .depth_stencil_format = sdl.c.SDL_GPU_TEXTUREFORMAT_D16_UNORM,
            .has_depth_stencil_target = true,
        },
    };
    const main_gpu_pipeline = sdl.c.SDL_CreateGPUGraphicsPipeline(
        gpu_device,
        &main_gpu_pipeline_create_info,
    ) orelse {
        log.err("SDL_CreateGPUGraphicsPipeline: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    };
    defer sdl.c.SDL_ReleaseGPUGraphicsPipeline(gpu_device, main_gpu_pipeline);

    const present_color_target_descriptions = [_]sdl.c.SDL_GPUColorTargetDescription{.{
        .format = sdl.c.SDL_GetGPUSwapchainTextureFormat(gpu_device, window),
        .blend_state = .{},
    }};
    const present_gpu_pipeline_create_info = sdl.c.SDL_GPUGraphicsPipelineCreateInfo{
        .vertex_shader = vert_shader,
        .fragment_shader = frag_shader,
        .vertex_input_state = .{
            .vertex_buffer_descriptions = &vertex_buffer_descriptions[0],
            .num_vertex_buffers = vertex_buffer_descriptions.len,
            .vertex_attributes = &vertex_attributes[0],
            .num_vertex_attributes = vertex_attributes.len,
        },
        .primitive_type = sdl.c.SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .rasterizer_state = .{},
        .multisample_state = .{},
        .depth_stencil_state = .{},
        .target_info = .{
            .color_target_descriptions = &present_color_target_descriptions[0],
            .num_color_targets = present_color_target_descriptions.len,
        },
    };
    const present_gpu_pipeline = sdl.c.SDL_CreateGPUGraphicsPipeline(
        gpu_device,
        &present_gpu_pipeline_create_info,
    ) orelse {
        log.err("SDL_CreateGPUGraphicsPipeline: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    };
    defer sdl.c.SDL_ReleaseGPUGraphicsPipeline(gpu_device, present_gpu_pipeline);

    sdl.c.SDL_ReleaseGPUShader(gpu_device, frag_shader);
    sdl.c.SDL_ReleaseGPUShader(gpu_device, vert_shader);

    const main_texture = sdl.c.SDL_CreateGPUTexture(gpu_device, &.{
        .type = sdl.c.SDL_GPU_TEXTURETYPE_2D,
        .format = sdl.c.SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .usage = sdl.c.SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | sdl.c.SDL_GPU_TEXTUREUSAGE_SAMPLER,
        .width = 800,
        .height = 600,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = sdl.c.SDL_GPU_SAMPLECOUNT_1,
    }) orelse {
        log.err("SDL_CreateGPUTexture: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    };
    defer sdl.c.SDL_ReleaseGPUTexture(gpu_device, main_texture);

    const main_depth_texture = sdl.c.SDL_CreateGPUTexture(gpu_device, &.{
        .type = sdl.c.SDL_GPU_TEXTURETYPE_2D,
        .format = sdl.c.SDL_GPU_TEXTUREFORMAT_D16_UNORM,
        .usage = sdl.c.SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET,
        .width = 800,
        .height = 600,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = sdl.c.SDL_GPU_SAMPLECOUNT_1,
    }) orelse {
        log.err("SDL_CreateGPUTexture: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    };
    defer sdl.c.SDL_ReleaseGPUTexture(gpu_device, main_depth_texture);

    const gradient_texture = sdl.c.SDL_CreateGPUTexture(gpu_device, &.{
        .type = sdl.c.SDL_GPU_TEXTURETYPE_2D,
        .format = sdl.c.SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .usage = sdl.c.SDL_GPU_TEXTUREUSAGE_SAMPLER,
        .width = 2,
        .height = 2,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = sdl.c.SDL_GPU_SAMPLECOUNT_1,
    }) orelse {
        log.err("SDL_CreateGPUTexture: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    };
    defer sdl.c.SDL_ReleaseGPUTexture(gpu_device, gradient_texture);

    const vertex_buffer = sdl.c.SDL_CreateGPUBuffer(gpu_device, &.{
        .usage = sdl.c.SDL_GPU_BUFFERUSAGE_VERTEX,
        .size = 16 * 1024 * 1024,
    }) orelse {
        log.err("SDL_CreateGPUBuffer: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    };
    defer sdl.c.SDL_ReleaseGPUBuffer(gpu_device, vertex_buffer);

    const transfer_buffer = sdl.c.SDL_CreateGPUTransferBuffer(gpu_device, &.{
        .usage = sdl.c.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = 16 * 1024 * 1024,
    }) orelse {
        log.err("SDL_CreateGPUTransferBuffer: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    };
    defer sdl.c.SDL_ReleaseGPUTransferBuffer(gpu_device, transfer_buffer);

    const sampler = sdl.c.SDL_CreateGPUSampler(gpu_device, &.{
        .min_filter = sdl.c.SDL_GPU_FILTER_LINEAR,
        .mag_filter = sdl.c.SDL_GPU_FILTER_LINEAR,
        .address_mode_u = sdl.c.SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_v = sdl.c.SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
    }) orelse {
        log.err("SDL_CreateGPUSampler: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    };
    defer sdl.c.SDL_ReleaseGPUSampler(gpu_device, sampler);

    { // static texture init
        const command_buffer = sdl.c.SDL_AcquireGPUCommandBuffer(gpu_device) orelse {
            log.err("SDL_AcquireGPUCommandBuffer: {s}", .{sdl.c.SDL_GetError()});
            return error.Sdl;
        };

        const bytes: [*][4]u8 = @alignCast(@ptrCast(
            sdl.c.SDL_MapGPUTransferBuffer(gpu_device, transfer_buffer, true) orelse {
                log.err("SDL_MapGPUTransferBuffer: {s}", .{sdl.c.SDL_GetError()});
                return error.Sdl;
            },
        ));
        @memcpy(
            bytes,
            &[_][4]u8{
                .{ 0, 0, 0, 0xff },
                .{ 0xff, 0, 0, 0xff },
                .{ 0, 0xff, 0, 0xff },
                .{ 0xff, 0xff, 0, 0xff },
            },
        );
        sdl.c.SDL_UnmapGPUTransferBuffer(gpu_device, transfer_buffer);

        const copy_pass = sdl.c.SDL_BeginGPUCopyPass(command_buffer);
        sdl.c.SDL_UploadToGPUTexture(copy_pass, &.{
            .transfer_buffer = transfer_buffer,
            .offset = 0,
            .pixels_per_row = 2,
            .rows_per_layer = 2,
        }, &.{ .texture = gradient_texture, .mip_level = 0, .layer = 0, .x = 0, .y = 0, .z = 0, .w = 2, .h = 2, .d = 1 }, true);
        sdl.c.SDL_EndGPUCopyPass(copy_pass);

        if (!sdl.c.SDL_SubmitGPUCommandBuffer(command_buffer)) {
            log.err("SDL_SubmitGPUCommandBuffer: {s}", .{sdl.c.SDL_GetError()});
            return error.Sdl;
        }
    }

    var input = Input.init(gpa);
    try input.map.put(.{ .keyboard = sdl.c.SDL_SCANCODE_W }, .forward);
    try input.map.put(.{ .keyboard = sdl.c.SDL_SCANCODE_S }, .backward);
    try input.map.put(.{ .keyboard = sdl.c.SDL_SCANCODE_D }, .right);
    try input.map.put(.{ .keyboard = sdl.c.SDL_SCANCODE_A }, .left);
    try input.map.put(.{ .keyboard = sdl.c.SDL_SCANCODE_SPACE }, .up);
    try input.map.put(.{ .keyboard = sdl.c.SDL_SCANCODE_LCTRL }, .down);
    defer input.deinit();

    var state = try State.init(gpa);
    defer state.deinit();

    var frame_timer = try std.time.Timer.start();
    var lag: u64 = 0;

    frame_timer.reset();
    main_loop: while (true) {
        lag += @min(frame_timer.lap(), max_tick_ns);

        var event: sdl.c.SDL_Event = undefined;
        while (sdl.c.SDL_PollEvent(&event)) {
            if (event.type == sdl.c.SDL_EVENT_QUIT) break :main_loop;
            if (event.type == sdl.c.SDL_EVENT_KEY_DOWN) switch (event.key.key) {
                sdl.c.SDLK_ESCAPE => break :main_loop,
                else => {},
            };
            input.accumulate(event);
        }

        while (lag >= tick_ns) {

            // begin update
            state.camera.update(&input);
            // end update

            input.decay();
            lag -= tick_ns;
        }

        const alpha = @as(f32, @floatFromInt(lag)) / @as(f32, @floatFromInt(tick_ns));
        _ = alpha;

        // begin draw
        const command_buffer = sdl.c.SDL_AcquireGPUCommandBuffer(gpu_device) orelse {
            log.err("SDL_AcquireGPUCommandBuffer: {s}", .{sdl.c.SDL_GetError()});
            return error.Sdl;
        };

        const bytes: [*]Vertex = @alignCast(@ptrCast(
            sdl.c.SDL_MapGPUTransferBuffer(gpu_device, transfer_buffer, true) orelse {
                log.err("SDL_MapGPUTransferBuffer: {s}", .{sdl.c.SDL_GetError()});
                return error.Sdl;
            },
        ));
        @memcpy(
            bytes,
            &[_]Vertex{
                .{ .pos = .{ -1, -1, 0 }, .uv = .{ 0, 0 } },
                .{ .pos = .{ 1, -1, 0 }, .uv = .{ 1, 0 } },
                .{ .pos = .{ -1, 1, 0 }, .uv = .{ 0, 1 } },
                .{ .pos = .{ 1, 1, 0 }, .uv = .{ 1, 1 } },
                .{ .pos = .{ 1, -1, 0 }, .uv = .{ 1, 0 } },
                .{ .pos = .{ -1, 1, 0 }, .uv = .{ 0, 1 } },
                //
                .{ .pos = .{ -0.5, -0.5, 0 }, .uv = .{ 0, 0 } },
                .{ .pos = .{ -0.5, 0.5, 0 }, .uv = .{ 0, 1 } },
                .{ .pos = .{ 0.5, -0.5, 0 }, .uv = .{ 1, 0 } },
            },
        );
        sdl.c.SDL_UnmapGPUTransferBuffer(gpu_device, transfer_buffer);
        const copy_pass = sdl.c.SDL_BeginGPUCopyPass(command_buffer);
        sdl.c.SDL_UploadToGPUBuffer(copy_pass, &.{
            .transfer_buffer = transfer_buffer,
            .offset = 0,
        }, &.{
            .buffer = vertex_buffer,
            .offset = 0,
            .size = @sizeOf(Vertex) * 9, // NOTE
        }, true);
        sdl.c.SDL_EndGPUCopyPass(copy_pass);

        const main_color_target_infos = [_]sdl.c.SDL_GPUColorTargetInfo{.{
            .texture = main_texture,
        }};
        const main_render_pass = sdl.c.SDL_BeginGPURenderPass(
            command_buffer,
            &main_color_target_infos[0],
            main_color_target_infos.len,
            &.{
                .texture = main_depth_texture,
                .clear_depth = 1,
                .load_op = sdl.c.SDL_GPU_LOADOP_CLEAR,
                .store_op = sdl.c.SDL_GPU_STOREOP_STORE,
            },
        );
        sdl.c.SDL_BindGPUGraphicsPipeline(main_render_pass, main_gpu_pipeline);
        const main_vertex_buffers = [_]sdl.c.SDL_GPUBufferBinding{
            .{ .buffer = vertex_buffer, .offset = 0 },
        };
        sdl.c.SDL_BindGPUVertexBuffers(
            main_render_pass,
            0,
            &main_vertex_buffers[0],
            main_vertex_buffers.len,
        );
        sdl.c.SDL_BindGPUFragmentSamplers(main_render_pass, 0, &.{
            .texture = gradient_texture,
            .sampler = sampler,
        }, 1);
        sdl.c.SDL_DrawGPUPrimitives(main_render_pass, 3, 1, 6, 0);
        sdl.c.SDL_EndGPURenderPass(main_render_pass);

        var swapchain_texture: ?*sdl.c.SDL_GPUTexture = null;
        var swapchain_width: u32 = 0;
        var swapchain_height: u32 = 0;
        if (!sdl.c.SDL_WaitAndAcquireGPUSwapchainTexture(
            command_buffer,
            window,
            &swapchain_texture,
            &swapchain_width,
            &swapchain_height,
        )) {
            log.err("SDL_WaitAndAcquireGPUSwapchainTexture: {s}", .{sdl.c.SDL_GetError()});
            return error.Sdl;
        }
        const present_color_target_infos = [_]sdl.c.SDL_GPUColorTargetInfo{.{
            .texture = swapchain_texture,
        }};
        const present_render_pass = sdl.c.SDL_BeginGPURenderPass(
            command_buffer,
            &present_color_target_infos[0],
            present_color_target_infos.len,
            null,
        );
        sdl.c.SDL_BindGPUGraphicsPipeline(present_render_pass, present_gpu_pipeline);
        const present_vertex_buffers = [_]sdl.c.SDL_GPUBufferBinding{
            .{ .buffer = vertex_buffer, .offset = 0 },
        };
        sdl.c.SDL_BindGPUVertexBuffers(
            present_render_pass,
            0,
            &present_vertex_buffers[0],
            present_vertex_buffers.len,
        );
        sdl.c.SDL_BindGPUFragmentSamplers(present_render_pass, 0, &.{
            .texture = main_texture,
            .sampler = sampler,
        }, 1);
        sdl.c.SDL_DrawGPUPrimitives(present_render_pass, 6, 1, 0, 0);
        sdl.c.SDL_EndGPURenderPass(present_render_pass);

        if (!sdl.c.SDL_SubmitGPUCommandBuffer(command_buffer)) {
            log.err("SDL_SubmitGPUCommandBuffer: {s}", .{sdl.c.SDL_GetError()});
            return error.Sdl;
        }
        // end draw
    }
}

const State = struct {
    camera: Camera,
    boxes: std.ArrayList(Box),

    fn init(gpa: std.mem.Allocator) !State {
        var state = State{
            .camera = .{
                .pos = .origin,
                .facing = .{ .data = .{ 1.0, 0.0, 0.0, 0.0 } },
                .yaw = 0.0,
                .pitch = 0.0,
            },
            .boxes = .init(gpa),
        };

        try state.boxes.append(.{
            .low = .{ .data = .{ -1, -1, -1, 1 } },
            .high = .{ .data = .{ 1, 1, 1, 1 } },
        });

        return state;
    }

    fn deinit(state: *State) void {
        state.boxes.deinit();
        state.* = undefined;
    }
};

const Camera = struct {
    pos: math.Vec,
    facing: math.Vec,
    yaw: f32,
    pitch: f32,

    const up = math.Vec{ .data = .{ 0.0, 1.0, 0.0, 0.0 } };
    const mouse_sensitivity = 0.002;
    const move_speed = 0.2;

    fn update(camera: *Camera, input: *Input) void {
        camera.yaw += input.mouse_delta.y * mouse_sensitivity;
        camera.pitch += input.mouse_delta.x * mouse_sensitivity;
        camera.pitch = std.math.clamp(camera.pitch, -0.49 * std.math.pi, 0.49 * std.math.pi);

        camera.facing = math.normalize(math.Vec{ .data = .{
            @cos(camera.pitch) * @cos(camera.yaw),
            @sin(camera.pitch),
            @cos(camera.pitch) * @sin(camera.yaw),
            0.0,
        } });

        const forward = math.normalize(math.Vec{ .data = .{
            @cos(camera.yaw),
            0.0,
            @sin(camera.yaw),
            0.0,
        } });

        const right = math.Vec{ .data = .{
            @cos(camera.yaw + 0.5 * std.math.pi),
            0.0,
            @sin(camera.yaw + 0.5 * std.math.pi),
            0.0,
        } };

        if (input.peek(.forward).held) camera.pos = math.add(camera.pos, forward);
        if (input.peek(.backward).held) camera.pos = math.sub(camera.pos, forward);
        if (input.peek(.right).held) camera.pos = math.add(camera.pos, right);
        if (input.peek(.left).held) camera.pos = math.sub(camera.pos, right);
        if (input.peek(.up).held) camera.pos = math.add(camera.pos, up);
        if (input.peek(.down).held) camera.pos = math.sub(camera.pos, up);
    }
};

const Box = struct {
    low: math.Vec,
    high: math.Vec,
};

const Vertex = extern struct {
    pos: [3]f32,
    uv: [2]f32,
};
