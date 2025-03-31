const std = @import("std");

const sdl = @import("sdl.zig");
const zm = @import("zmath");

const Input = @import("input.zig");
const SoundSystem = @import("sound_system.zig");

const SpaceBuilder = @import("raytracer.zig").Builder;

const log = std.log;

pub const ticks_per_second = 83;
pub const tick: f32 = 1.0 / @as(f32, @floatFromInt(ticks_per_second));
pub const tick_ns: u64 = 1000_000_000 / ticks_per_second;
pub const max_tick_ns: u64 = @intFromFloat(0.1 * 1e9);

pub fn main() !void {
    var gpa_struct: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_struct.deinit();
    const gpa = gpa_struct.allocator();

    sdl.c.SDL_SetMainReady();

    if (!sdl.c.SDL_Init(sdl.c.SDL_INIT_VIDEO | sdl.c.SDL_INIT_AUDIO)) {
        log.err("SDL_Init: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    }
    defer sdl.c.SDL_Quit();

    const window = sdl.c.SDL_CreateWindow("audio-raytracing", 800, 600, 0) orelse {
        log.err("SDL_CreateWindow: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    };
    defer sdl.c.SDL_DestroyWindow(window);

    if (!sdl.c.SDL_SetWindowRelativeMouseMode(window, true)) {
        log.err("SDL_SetWindowRelativeMouseMode: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    }

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
            .num_uniform_buffers = 1,
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
        }, &.{
            .texture = gradient_texture,
            .mip_level = 0,
            .layer = 0,
            .x = 0,
            .y = 0,
            .z = 0,
            .w = 2,
            .h = 2,
            .d = 1,
        }, true);
        sdl.c.SDL_EndGPUCopyPass(copy_pass);

        if (!sdl.c.SDL_SubmitGPUCommandBuffer(command_buffer)) {
            log.err("SDL_SubmitGPUCommandBuffer: {s}", .{sdl.c.SDL_GetError()});
            return error.Sdl;
        }
    }

    var vertices = std.ArrayList(Vertex).init(gpa);
    defer vertices.deinit();
    { // read all triangles from obj file
        var raw_vertices = std.ArrayList([3]f32).init(gpa);
        defer raw_vertices.deinit();
        var raw_uvs = std.ArrayList([2]f32).init(gpa);
        defer raw_uvs.deinit();
        var triangles = std.ArrayList(
            struct { a: u32, b: u32, c: u32, d: u32, e: u32, f: u32 },
        ).init(gpa);
        defer triangles.deinit();

        const file = try std.fs.cwd().openFile(
            "data/world.obj",
            .{ .mode = .read_only },
        );
        defer file.close();
        const bytes = try file.reader().readAllAlloc(gpa, 1_000_000);
        defer gpa.free(bytes);

        var it = std.mem.tokenizeAny(u8, bytes, "\n");
        while (it.next()) |line| {
            if (std.mem.eql(u8, line[0..2], "vt")) {
                std.debug.print("tex\t{s}\n", .{line});
                var it2 = std.mem.tokenizeAny(u8, line, " ");
                _ = it2.next();
                try raw_uvs.append(.{
                    try std.fmt.parseFloat(f32, it2.next().?),
                    try std.fmt.parseFloat(f32, it2.next().?),
                });
            } else if (std.mem.eql(u8, line[0..1], "v")) {
                std.debug.print("vertex\t{s}\n", .{line});
                var it2 = std.mem.tokenizeAny(u8, line, " ");
                _ = it2.next();
                try raw_vertices.append(.{
                    try std.fmt.parseFloat(f32, it2.next().?),
                    try std.fmt.parseFloat(f32, it2.next().?),
                    try std.fmt.parseFloat(f32, it2.next().?),
                });
            } else if (std.mem.eql(u8, line[0..1], "f")) {
                std.debug.print("face\t{s}\n", .{line});
                var it2 = std.mem.tokenizeAny(u8, line, " /");
                _ = it2.next();
                try triangles.append(.{
                    .a = try std.fmt.parseInt(u32, it2.next().?, 10) - 1,
                    .b = try std.fmt.parseInt(u32, it2.next().?, 10) - 1,
                    .c = try std.fmt.parseInt(u32, it2.next().?, 10) - 1,
                    .d = try std.fmt.parseInt(u32, it2.next().?, 10) - 1,
                    .e = try std.fmt.parseInt(u32, it2.next().?, 10) - 1,
                    .f = try std.fmt.parseInt(u32, it2.next().?, 10) - 1,
                });
            }
        }

        for (triangles.items) |tri| {
            try vertices.append(.{
                .pos = raw_vertices.items[tri.a],
                .uv = raw_uvs.items[tri.b],
            });
            try vertices.append(.{
                .pos = raw_vertices.items[tri.c],
                .uv = raw_uvs.items[tri.d],
            });
            try vertices.append(.{
                .pos = raw_vertices.items[tri.e],
                .uv = raw_uvs.items[tri.f],
            });
        }
    }

    var space_builder = SpaceBuilder(u32).init(gpa);
    {
        var i: usize = 0;
        std.debug.assert(vertices.items.len % 3 == 0);
        while (i < vertices.items.len) : (i += 3) {
            const v0 = vertices.items[i];
            const v1 = vertices.items[i + 1];
            const v2 = vertices.items[i + 2];
            const low: [3]f32 = .{
                @min(v0.pos[0], @min(v1.pos[0], v2.pos[0])),
                @min(v0.pos[1], @min(v1.pos[1], v2.pos[1])),
                @min(v0.pos[2], @min(v1.pos[2], v2.pos[2])),
            };
            const high: [3]f32 = .{
                @max(v0.pos[0], @max(v1.pos[0], v2.pos[0])),
                @max(v0.pos[1], @max(v1.pos[1], v2.pos[1])),
                @max(v0.pos[2], @max(v1.pos[2], v2.pos[2])),
            };
            try space_builder.add(low, high, @intCast(i));
        }
    }
    var space = try space_builder.finish();
    defer space.deinit();

    var input = Input.init(gpa);
    try input.map.put(.{ .keyboard = sdl.c.SDL_SCANCODE_W }, .forward);
    try input.map.put(.{ .keyboard = sdl.c.SDL_SCANCODE_S }, .backward);
    try input.map.put(.{ .keyboard = sdl.c.SDL_SCANCODE_D }, .right);
    try input.map.put(.{ .keyboard = sdl.c.SDL_SCANCODE_A }, .left);
    try input.map.put(.{ .keyboard = sdl.c.SDL_SCANCODE_SPACE }, .up);
    try input.map.put(.{ .keyboard = sdl.c.SDL_SCANCODE_LCTRL }, .down);
    try input.map.put(.{ .mouse = sdl.c.SDL_BUTTON_LEFT }, .fire);
    defer input.deinit();

    const sound_system = try SoundSystem.init(gpa);
    defer sound_system.deinit();

    const music = try sound_system.loadSound("data/sounds/space_cadet_training_montage.wav");
    const music_handle = try sound_system.playSound(.{
        .sound = music,
        .pos = zm.f32x4(0.0, 0.0, 0.0, 0.0),
        .repeat = true,
        .gain = 1.0,
    });
    _ = music_handle;

    const blip = try sound_system.loadSound("data/sounds/blipSelect.wav");

    var state = try State.init(gpa);
    defer state.deinit();

    var frame_timer = try std.time.Timer.start();
    var lag: u64 = 0;
    var time: f64 = 0.0;

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

            if (input.peek(.fire).pressed) {
                _ = try sound_system.playSound(.{ .sound = blip, .pos = zm.f32x4s(0.0), .gain = 0.2 });
            }

            input.decay();
            // end update

            lag -= tick_ns;
            time += 1.0 / @as(f64, @floatFromInt(ticks_per_second));
        }

        // begin audio state update here maybe?
        // std.debug.print("{}\n", .{state.camera});
        // instead of locking, a triple buffer mailbox would prevent audio glitches
        {
            sound_system.mutex.lock();
            defer sound_system.mutex.unlock();
            var t = std.time.Timer.start() catch unreachable;
            defer std.debug.print("{d:.2}\n", .{@as(f64, @floatFromInt(t.lap())) * 1e-6});

            sound_system.listener = state.camera.pos;
            sound_system.orientation = zm.quatFromRollPitchYaw(
                state.camera.pitch,
                state.camera.yaw,
                0,
            );

            var harmonic_mean_dist: f32 = 0;
            var capped_mean_dist: f32 = 0;

            for (raycast_sphere_pattern) |dir| {
                const src: [3]f32 = .{
                    state.camera.pos[0],
                    state.camera.pos[1],
                    state.camera.pos[2],
                };
                const isects, const n = space.raycastCapacity(src, dir, 128);
                var best: u32 = std.math.maxInt(u32);
                var dist: f32 = std.math.inf(f32);

                // find the closest intersecting triangle
                for (isects[0..n]) |i| {
                    const d, _ = rayTriangleIntersection(
                        src,
                        dir,
                        vertices.items[i].pos,
                        vertices.items[i + 1].pos,
                        vertices.items[i + 2].pos,
                    ) orelse continue;
                    if (d < dist) {
                        dist = d;
                        best = @intCast(i);
                    }
                }
                std.debug.print("{} {}\n", .{ best, dist });

                harmonic_mean_dist += 1.0 / dist;
                capped_mean_dist += @min(dist, 25.0);
            }
            harmonic_mean_dist =
                @as(f32, @floatFromInt(raycast_sphere_pattern.len)) / harmonic_mean_dist;
            capped_mean_dist /= @as(f32, @floatFromInt(raycast_sphere_pattern.len));

            std.debug.print(
                "distance: {}\n          {}\n",
                .{ harmonic_mean_dist, capped_mean_dist },
            );

            var it = sound_system.playing.iterator();
            while (it.next()) |p| {
                p.value_ptr.reverb.feedback_gain =
                    @sqrt((26.0 - capped_mean_dist) / 25.0);
                p.value_ptr.wet =
                    ((50.0 - capped_mean_dist) / 50.0) * ((50.0 - capped_mean_dist) / 50.0);
            }
        }
        // end audio state update

        // begin draw
        const alpha = @as(f32, @floatFromInt(lag)) / @as(f32, @floatFromInt(tick_ns));

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
                .{ .pos = .{ -1, 1, 0 }, .uv = .{ 0, 0 } },
                .{ .pos = .{ 1, 1, 0 }, .uv = .{ 1, 0 } },
                .{ .pos = .{ 1, -1, 0 }, .uv = .{ 1, 1 } },
                .{ .pos = .{ -1, 1, 0 }, .uv = .{ 0, 0 } },
                .{ .pos = .{ 1, -1, 0 }, .uv = .{ 1, 1 } },
                .{ .pos = .{ -1, -1, 0 }, .uv = .{ 0, 1 } },
            }, // quad for backbuffer -> swapchain renderpass
        );
        @memcpy(
            bytes + 6,
            vertices.items,
        );
        sdl.c.SDL_UnmapGPUTransferBuffer(gpu_device, transfer_buffer);
        const copy_pass = sdl.c.SDL_BeginGPUCopyPass(command_buffer);
        sdl.c.SDL_UploadToGPUBuffer(copy_pass, &.{
            .transfer_buffer = transfer_buffer,
            .offset = 0,
        }, &.{
            .buffer = vertex_buffer,
            .offset = 0,
            .size = @sizeOf(Vertex) * (6 + @as(u32, @intCast(vertices.items.len))),
        }, true);
        sdl.c.SDL_EndGPUCopyPass(copy_pass);

        const main_color_target_infos = [_]sdl.c.SDL_GPUColorTargetInfo{.{
            .texture = main_texture,
            .load_op = sdl.c.SDL_GPU_LOADOP_CLEAR,
            .store_op = sdl.c.SDL_GPU_STOREOP_STORE,
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
        sdl.c.SDL_PushGPUVertexUniformData(
            command_buffer,
            0,
            &state.camera.vp(alpha),
            @sizeOf(zm.Mat),
        );
        sdl.c.SDL_DrawGPUPrimitives(main_render_pass, @intCast(vertices.items.len), 1, 6, 0);
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
        sdl.c.SDL_PushGPUVertexUniformData(
            command_buffer,
            0,
            &zm.identity(),
            @sizeOf(zm.Mat),
        );
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

    fn init(gpa: std.mem.Allocator) !State {
        _ = gpa;
        return State{
            .camera = .{
                .pos = zm.f32x4(0.0, 0.0, 0.0, 1.0),
                .yaw = 0.0,
                .pitch = 0.0,
                .prev_pos = zm.f32x4(0.0, 0.0, 0.0, 1.0),
                .prev_yaw = 0.0,
                .prev_pitch = 0.0,
            },
        };
    }

    fn deinit(state: *State) void {
        state.* = undefined;
    }
};

const Camera = struct {
    pos: zm.Vec,
    yaw: f32,
    pitch: f32,

    prev_pos: zm.Vec,
    prev_yaw: f32,
    prev_pitch: f32,

    const up = zm.f32x4(0.0, 1.0, 0.0, 0.0);
    const mouse_sensitivity = 0.3;
    const move_speed = 5;

    fn update(camera: *Camera, input: *Input) void {
        camera.prev_pos = camera.pos;
        camera.prev_yaw = camera.yaw;
        camera.prev_pitch = camera.pitch;

        camera.yaw += input.mouse_delta[0] * mouse_sensitivity * tick;
        camera.pitch -= input.mouse_delta[1] * mouse_sensitivity * tick;
        camera.pitch = std.math.clamp(camera.pitch, -0.49 * std.math.pi, 0.49 * std.math.pi);

        const forward = zm.f32x4(
            @cos(camera.yaw),
            0.0,
            @sin(camera.yaw),
            0.0,
        ) * zm.f32x4s(move_speed * tick);

        const right = zm.f32x4(
            @cos(camera.yaw + 0.5 * std.math.pi),
            0.0,
            @sin(camera.yaw + 0.5 * std.math.pi),
            0.0,
        ) * zm.f32x4s(move_speed * tick);

        if (input.peek(.forward).held) camera.pos += forward;
        if (input.peek(.backward).held) camera.pos -= forward;
        if (input.peek(.right).held) camera.pos += right;
        if (input.peek(.left).held) camera.pos -= right;
        if (input.peek(.up).held) camera.pos += up * zm.f32x4s(move_speed * tick);
        if (input.peek(.down).held) camera.pos -= up * zm.f32x4s(move_speed * tick);
    }

    fn vp(camera: Camera, alpha: f32) zm.Mat {
        // NOTE tbh the delay is somewhat noticable even at a higher tickrate
        // consider using deltatime updates specifically for mouse camera
        const pos = zm.lerp(camera.prev_pos, camera.pos, alpha);
        const yaw = (1 - alpha) * camera.prev_yaw + alpha * camera.yaw;
        const pitch = (1 - alpha) * camera.prev_pitch + alpha * camera.pitch;

        const facing = zm.normalize3(zm.f32x4(
            @cos(pitch) * @cos(yaw),
            @sin(pitch),
            @cos(pitch) * @sin(yaw),
            0.0,
        ));

        return zm.mul(
            zm.lookAtRh(pos, pos + facing, up),
            zm.perspectiveFovRh(std.math.degreesToRadians(69), 4.0 / 3.0, 0.01, 100.0),
        );
    }
};

const Box = struct {
    low: zm.Vec,
    high: zm.Vec,
};

const Vertex = extern struct {
    pos: [3]f32,
    uv: [2]f32,
};

fn rayTriangleIntersection(
    _src: [3]f32,
    _dir: [3]f32,
    v0: [3]f32,
    v1: [3]f32,
    v2: [3]f32,
) ?struct { f32, bool } {
    const src = zm.loadArr3(_src);
    const dir = zm.normalize3(zm.loadArr3(_dir));
    const a = zm.loadArr3(v0);
    const b = zm.loadArr3(v1);
    const c = zm.loadArr3(v2);

    // std.debug.print("{}->{}\n{} {} {}\n", .{ src, dir, a, b, c });

    const eps: f32 = 1e-6;

    const ab = b - a;
    const ac = c - a;

    const h = zm.cross3(dir, ac);
    const d = zm.dot3(ab, h);
    // std.debug.print("1 {}\n", .{d});
    if (d[0] > -eps and d[0] < eps) return null;

    const f = zm.f32x4s(1.0) / d;
    const s = src - a;
    const u = f * zm.dot3(s, h);
    // std.debug.print("2 {}\n", .{u});
    if (u[0] < 0.0 or u[0] > 1.0) return null;

    const q = zm.cross3(s, ab);
    const v = f * zm.dot3(dir, q);
    // std.debug.print("3\n", .{});
    if (v[0] < 0.0 or u[0] + v[0] > 1.0) return null;

    const t = f * zm.dot3(ac, q);
    // std.debug.print("4\n", .{});
    if (t[0] < eps) return null;

    // std.debug.print("5\n", .{});
    return .{
        t[0],
        undefined, // TODO check if we hit front/back using winding order
    };
}

const raycast_sphere_pattern = [_][3]f32{
    .{ 1, 0, 0 },
    .{ -1, 0, 0 },
    .{ 0, 1, 0 },
    .{ 0, -1, 0 },
    .{ 0, 0, 1 },
    .{ 0, 0, -1 },
    .{ 1, 1, 1 },
    .{ 1, 1, -1 },
    .{ 1, -1, 1 },
    .{ 1, -1, -1 },
    .{ -1, 1, 1 },
    .{ -1, 1, -1 },
    .{ -1, -1, 1 },
    .{ -1, -1, -1 },
};
