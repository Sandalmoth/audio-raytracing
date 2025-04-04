const std = @import("std");

const sdl = @import("sdl.zig");
const zm = @import("zmath");

const log = std.log.scoped(.sound_system);

const SoundSystem = @This();

/// sound effects (any sound to go into the spatializer) should have this format
const sound_effect_spec = sdl.c.SDL_AudioSpec{
    .format = sdl.c.SDL_AUDIO_F32,
    .channels = 1,
    .freq = 44100,
};
/// the spiatializer will then output sound in this format
const sound_render_spec = sdl.c.SDL_AudioSpec{
    .format = sdl.c.SDL_AUDIO_F32,
    .channels = 2,
    .freq = 44100,
};

const hrtf: struct {
    irs_l: []const []const f32,
    irs_r: []const []const f32,
} = @import("hrtf.zon");

const frame_size = 128;
const speed_of_sound = 350.0;

gpa: std.mem.Allocator,
sounds: std.ArrayListUnmanaged(Sound),
playing: std.AutoArrayHashMapUnmanaged(usize, Playing),
playing_counter: usize,
stream: *sdl.c.SDL_AudioStream,
listener: zm.Vec,
orientation: zm.Quat,
stereo_frame_buffer: [2 * frame_size][2]f32,
mutex: std.Thread.Mutex,

pub fn init(gpa: std.mem.Allocator) !*SoundSystem {
    const system = try gpa.create(SoundSystem);
    errdefer gpa.destroy(system);

    system.gpa = gpa;
    system.sounds = .empty;
    system.playing = .empty;
    system.playing_counter = 0;
    system.listener = zm.f32x4s(0.0);
    system.orientation = zm.qidentity();
    system.stereo_frame_buffer = std.mem.zeroes([2 * frame_size][2]f32);
    system.mutex = .{};
    system.stream = sdl.c.SDL_OpenAudioDeviceStream(
        sdl.c.SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK,
        null,
        null,
        null,
    ) orelse {
        log.err("SDL_OpenAudioDeviceStream: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    };
    errdefer sdl.c.SDL_DestroyAudioStream(system.stream);
    if (!sdl.c.SDL_SetAudioStreamFormat(system.stream, &sound_render_spec, null)) {
        log.err("SDL_SetAudioStreamFormat: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    }

    var audio_stream_input_format: sdl.c.SDL_AudioSpec = undefined;
    var audio_stream_output_format: sdl.c.SDL_AudioSpec = undefined;
    if (!sdl.c.SDL_GetAudioStreamFormat(
        system.stream,
        &audio_stream_input_format,
        &audio_stream_output_format,
    )) {
        log.err("SDL_GetAudioStreamFormat: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    }
    if (!sdl.c.SDL_ResumeAudioStreamDevice(system.stream)) {
        log.err("SDL_ResumeAudioStreamDevice: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    }

    if (!sdl.c.SDL_SetAudioStreamGetCallback(system.stream, callback, system)) {
        log.err("SDL_SetAudioStreamGetCallback: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    }

    return system;
}

pub fn deinit(system: *SoundSystem) void {
    sdl.c.SDL_DestroyAudioStream(system.stream);
    for (system.sounds.items) |*sound| sound.deinit();
    system.sounds.deinit(system.gpa);
    system.playing.deinit(system.gpa);
    system.gpa.destroy(system);
}

// NOTE automatically freed when the system is freed
pub fn loadSound(system: *SoundSystem, filename: [*c]const u8) !usize {
    try system.sounds.ensureUnusedCapacity(system.gpa, 1);
    const sound = try Sound.init(filename);
    const result = system.sounds.items.len;
    system.sounds.appendAssumeCapacity(sound);
    return result;
}

pub fn playSound(system: *SoundSystem, p: Playing) !usize {
    system.mutex.lock();
    defer system.mutex.unlock();
    try system.playing.ensureUnusedCapacity(system.gpa, 1);
    const result = system.playing_counter;
    system.playing.putAssumeCapacity(result, p);
    system.playing_counter += 1;
    return result;
}

fn callback(
    ctx: ?*anyopaque,
    stream: ?*sdl.c.SDL_AudioStream,
    additional_amount: c_int,
    total_amount: c_int,
) callconv(.c) void {
    const system = @as(?*SoundSystem, @alignCast(@ptrCast(ctx))).?;
    system.mutex.lock();
    defer system.mutex.unlock();
    // var t = std.time.Timer.start() catch unreachable;
    // defer std.debug.print("{d:.2}\n", .{@as(f64, @floatFromInt(t.lap())) * 1e-6});

    var n_samples = @divTrunc(additional_amount, 2 * @sizeOf(f32));
    _ = total_amount;

    var frame_index: usize = 0;
    const total_frames: usize = @as(usize, @intCast(@divTrunc(n_samples, 128))) + 1;
    while (n_samples > 0) : (n_samples -= 128) {
        var ambisonic = std.mem.zeroes([4][frame_size]f32);
        var reverb = std.mem.zeroes([frame_size]f32); // could be a temporary inside buildAR
        system.buildAmbisonicReverb(&ambisonic, &reverb, frame_index, total_frames);
        system.rotateAmbisonic(&ambisonic);
        system.ambisonicToStereo(ambisonic);
        if (!sdl.c.SDL_PutAudioStreamData(
            stream,
            &system.stereo_frame_buffer[0],
            2 * frame_size * @sizeOf(f32),
        )) {
            log.err("SDL_PutAudioStreamData: {s}", .{sdl.c.SDL_GetError()});
        }

        for (0..frame_size) |i| {
            system.stereo_frame_buffer[i] = system.stereo_frame_buffer[i + frame_size];
            system.stereo_frame_buffer[i + frame_size] = .{ 0.0, 0.0 };
        }
        frame_index += 1;
    }

    // remove any sounds that finished playing
    const values = system.playing.values();
    var i = values.len;
    while (i > 0) : (i -= 1) {
        if (values[i - 1].finished) {
            system.playing.swapRemoveAt(i - 1);
        }
    }
}

fn buildAmbisonicReverb(
    system: *SoundSystem,
    buf: *[4][frame_size]f32,
    buf2: *[frame_size]f32,
    frame_index: usize,
    total_frames: usize,
) void {
    // buf - ambisonic
    // buf2 - reverb
    var it = system.playing.iterator();
    while (it.next()) |kv| {
        const p = kv.value_ptr;
        // ambisonic components
        const sh: [4]f32, const dist: f32 = blk: {
            // if distance is below threshold, smoothly scale to 0 directional components
            // we are using N3D normalization for the spherical harmonics
            // however, the coordinate system is rotated to match
            // +x -> front
            // +y -> up
            // +z -> right
            const dir = p.pos - system.listener; // from listener towards source
            const len = zm.length3(dir)[0];
            const t: f32 = 0.1;
            const norm = if (len < 1e-6)
                0.0
            else if (len < t)
                @sqrt(3.0) / t
            else
                @sqrt(3.0) / len;
            break :blk .{ .{
                1.0,
                norm * dir[0],
                norm * dir[1],
                norm * dir[2],
            }, len };
        };
        if (p.prev_dist == null) p.prev_dist = dist;
        if (p.reflections.x_pos_dist_prev == null)
            p.reflections.x_pos_dist_prev = p.reflections.x_pos_dist;
        if (p.reflections.x_neg_dist_prev == null)
            p.reflections.x_neg_dist_prev = p.reflections.x_neg_dist;
        if (p.reflections.y_pos_dist_prev == null)
            p.reflections.y_pos_dist_prev = p.reflections.y_pos_dist;
        if (p.reflections.y_neg_dist_prev == null)
            p.reflections.y_neg_dist_prev = p.reflections.y_neg_dist;
        if (p.reflections.z_pos_dist_prev == null)
            p.reflections.z_pos_dist_prev = p.reflections.z_pos_dist;
        if (p.reflections.z_neg_dist_prev == null)
            p.reflections.z_neg_dist_prev = p.reflections.z_neg_dist;
        const s = system.sounds.items[p.sound];
        const samples = s.samples();

        p.attenuation_eq.gains = std.math.clamp(
            @as(@Vector(4, f32), @splat(1.0)) -
                @as(@Vector(4, f32), @splat(1e-5 * dist)) * Equalizer.freqs,
            @as(@Vector(4, f32), @splat(0.0)),
            @as(@Vector(4, f32), @splat(1.0)),
        ); // air absorbtion
        // std.debug.print("{}\n", .{p.attenuation_eq.gains});
        p.attenuation_eq.gains *= @splat(1 / (dist + 1)); // distance attenuation

        var reverb_input = std.mem.zeroes([frame_size]f32);

        if (p.repeat) {
            for (0..128) |i| {
                {
                    const d = std.math.lerp(
                        p.prev_dist.?,
                        dist,
                        @as(f32, @floatFromInt(128 * frame_index + i)) /
                            @as(f32, @floatFromInt(128 * total_frames)),
                    );
                    // std.debug.print("{} {} {}\n", .{ frame_index, total_frames, d });
                    const foff = @as(f32, @floatFromInt(p.cursor)) +
                        4 * @as(f32, @floatFromInt(samples.len)) -
                        44100 * d / speed_of_sound;
                    const ioff = @as(usize, @intFromFloat(foff));
                    const beta = foff - @trunc(foff);
                    const sample = std.math.lerp(
                        samples[(ioff + i) % samples.len],
                        samples[(ioff + i - 1) % samples.len],
                        beta,
                    );
                    reverb_input[i] = sample * p.gain / (dist + 2);
                    for (0..4) |j| buf[j][i] += sh[j] * p.attenuation_eq.apply(sample) * p.gain;
                }

                // reflections go here
                // var offset: usize = 0;
                var sample: f32 = undefined;
                sample = dopplerReflectionResample(
                    p,
                    p.reflections.x_pos_dist_prev.?,
                    p.reflections.x_pos_dist,
                    frame_index,
                    i,
                    total_frames,
                    samples,
                ) * p.gain * p.reflections.x_pos_lam / (p.reflections.x_pos_dist + 1);
                buf[0][i] += sample;
                buf[1][i] += sample;
                sample = dopplerReflectionResample(
                    p,
                    p.reflections.x_neg_dist_prev.?,
                    p.reflections.x_neg_dist,
                    frame_index,
                    i,
                    total_frames,
                    samples,
                ) * p.gain * p.reflections.x_neg_lam / (p.reflections.x_neg_dist + 1);
                buf[0][i] += sample;
                buf[1][i] -= sample;
                sample = dopplerReflectionResample(
                    p,
                    p.reflections.y_pos_dist_prev.?,
                    p.reflections.y_pos_dist,
                    frame_index,
                    i,
                    total_frames,
                    samples,
                ) * p.gain * p.reflections.y_pos_lam / (p.reflections.y_pos_dist + 1);
                buf[0][i] += sample;
                buf[2][i] += sample;
                sample = dopplerReflectionResample(
                    p,
                    p.reflections.y_neg_dist_prev.?,
                    p.reflections.y_neg_dist,
                    frame_index,
                    i,
                    total_frames,
                    samples,
                ) * p.gain * p.reflections.y_neg_lam / (p.reflections.y_neg_dist + 1);
                buf[0][i] += sample;
                buf[2][i] -= sample;
                sample = dopplerReflectionResample(
                    p,
                    p.reflections.z_pos_dist_prev.?,
                    p.reflections.z_pos_dist,
                    frame_index,
                    i,
                    total_frames,
                    samples,
                ) * p.gain * p.reflections.z_pos_lam / (p.reflections.z_pos_dist + 1);
                buf[0][i] += sample;
                buf[3][i] += sample;
                sample = dopplerReflectionResample(
                    p,
                    p.reflections.z_neg_dist_prev.?,
                    p.reflections.z_neg_dist,
                    frame_index,
                    i,
                    total_frames,
                    samples,
                ) * p.gain * p.reflections.z_neg_lam / (p.reflections.z_neg_dist + 1);
                buf[0][i] += sample;
                buf[3][i] -= sample;
            }
            p.cursor += 128;
        } else {
            const begin = @min(p.cursor, samples.len);
            const end = @min(p.cursor + 128, samples.len);
            for (begin..end, 0..) |i, k| {
                reverb_input[k] = samples[i] * p.gain;
                for (0..4) |j| {
                    const sample = p.attenuation_eq.apply(
                        samples[i],
                    );
                    buf[j][i - p.cursor] += sh[j] * sample * p.gain / (dist + 2);
                }

                // reflections go here
            }
            p.cursor += 128;
            if (p.cursor >= samples.len + 65536) p.finished = true;
        }

        p.reverb.apply(reverb_input, buf2);

        // apply reverb nondirectionally
        for (0..frame_size) |i| {
            buf[0][i] += p.wet * buf2[i];
        }

        if (frame_index + 1 == total_frames) {
            p.prev_dist = dist;
            p.reflections.x_pos_dist_prev = p.reflections.x_pos_dist;
            p.reflections.x_neg_dist_prev = p.reflections.x_neg_dist;
            p.reflections.y_pos_dist_prev = p.reflections.y_pos_dist;
            p.reflections.y_neg_dist_prev = p.reflections.y_neg_dist;
            p.reflections.z_pos_dist_prev = p.reflections.z_pos_dist;
            p.reflections.z_neg_dist_prev = p.reflections.z_neg_dist;
        }
    }
}

fn rotateAmbisonic(system: *SoundSystem, ambisonic: *[4][frame_size]f32) void {
    for (0..frame_size) |i| {
        const a = zm.rotate(
            system.orientation,
            zm.f32x4(ambisonic[1][i], ambisonic[2][i], ambisonic[3][i], 1.0),
        );
        ambisonic[1][i] = a[0];
        ambisonic[2][i] = a[1];
        ambisonic[3][i] = a[2];
    }
}

const identity_ir = blk: {
    var ir = std.mem.zeroes([frame_size]f32);
    ir[0] = 1.0;
    break :blk ir;
};

fn ambisonicToStereo(system: *SoundSystem, ambisonic: [4][frame_size]f32) void {
    var conv_bufs: [2][2 * frame_size]f32 = undefined;
    for (0..4) |i| {
        convolve(&ambisonic[i], hrtf.irs_l[i], &conv_bufs[0]);
        convolve(&ambisonic[i], hrtf.irs_r[i], &conv_bufs[1]);
        for (0..2 * frame_size) |j| {
            system.stereo_frame_buffer[j][0] += conv_bufs[0][j];
            system.stereo_frame_buffer[j][1] += conv_bufs[1][j];
        }
    }
}

fn convolve(input: []const f32, ir: []const f32, output: []f32) void {
    std.debug.assert(output.len >= input.len + ir.len - 1);
    for (0..output.len) |i| output[i] = 0.0;
    for (0..input.len) |i| {
        for (0..ir.len) |j| {
            output[i + j] += input[i] * ir[j];
        }
    }
}

const Reflections = struct {
    x_pos_dist: f32 = 0.0,
    x_neg_dist: f32 = 0.0,
    x_pos_lam: f32 = 0.0,
    x_neg_lam: f32 = 0.0,
    y_pos_dist: f32 = 0.0,
    y_neg_dist: f32 = 0.0,
    y_pos_lam: f32 = 0.0,
    y_neg_lam: f32 = 0.0,
    z_pos_dist: f32 = 0.0,
    z_neg_dist: f32 = 0.0,
    z_pos_lam: f32 = 0.0,
    z_neg_lam: f32 = 0.0,

    x_pos_dist_prev: ?f32 = null,
    x_neg_dist_prev: ?f32 = null,
    y_pos_dist_prev: ?f32 = null,
    y_neg_dist_prev: ?f32 = null,
    z_pos_dist_prev: ?f32 = null,
    z_neg_dist_prev: ?f32 = null,
};

const Playing = struct {
    sound: usize, // should be a type safe id in a proper implementation
    pos: zm.Vec,
    prev_dist: ?f32 = null, // previous distance to listener
    gain: f32 = 1.0,
    cursor: usize = 0,
    repeat: bool = false,
    finished: bool = false,
    attenuation_eq: Equalizer = .{},
    reverb: Reverb = .init,
    wet: f32 = 0.00,
    reflections: Reflections = .{},
};

const Equalizer = struct {
    // breakpoints at 16 256 4096, LR2 filters
    const as: @Vector(4, f32) =
        .{ -0.9977229806593002, -0.9977229806593002, -0.9641755363925378, -0.5380310834985628 };
    const bs: [2]@Vector(4, f32) = .{
        .{ 0.0011385096703499323, 0.9988614903296501, 0.982087768196269, 0.7690155417492813 },
        .{ 0.0011385096703499323, -0.9988614903296501, -0.982087768196269, -0.7690155417492813 },
    };
    const freqs = @Vector(4, f32){ 4.0000e+00, 6.4000e+01, 1.0240e+03, 1.6384e+04 };

    gains: @Vector(4, f32) = @splat(1),
    zs: @Vector(4, f32) = @splat(0),

    fn apply(eq: *Equalizer, x: f32) f32 {
        // split into four bands using LR2 filters
        // 128hz, 768hz, 4608hz breakpoints
        const xs: @Vector(4, f32) = @splat(x);
        const ys = bs[0] * xs + eq.zs;
        eq.zs = bs[1] * xs - as * ys;
        // then apply a gain to each band and add back together
        // the select uses the low/high-pass filters to create four bands like so
        // low - mid_low,   mid_low - mid_high,   mid_high - high,   high
        const bands = ys - @shuffle(
            f32,
            ys,
            @as(@Vector(4, f32), @splat(0.0)),
            @as(@Vector(4, i32), .{ -1, 2, 3, -1 }),
        );
        // return @reduce(.Add, bands * eq.gains);
        return @reduce(.Add, bands * @Vector(4, f32){ 1, -1, 1, -1 } * eq.gains);
    }
};

const Reverb = struct {
    const diffuser_delays: [4][4]u32 = .{
        .{ 383, 947, 1489, 3571 },
        .{ 31, 449, 937, 2671 },
        .{ 131, 179, 1619, 1879 },
        .{ 463, 593, 443, 887 },
    };
    const diffuser_shuffles: [4][4]u32 = .{
        .{ 3, 1, 0, 2 },
        .{ 0, 1, 3, 2 },
        .{ 0, 1, 3, 2 },
        .{ 2, 0, 3, 1 },
    };
    const diffuser_polarities: [4][4]f32 = .{
        .{ -1, 1, 1, -1 },
        .{ -1, -1, 1, 1 },
        .{ 1, -1, 1, -1 },
        .{ -1, 1, 1, -1 },
    };
    const feedback_delays: [4]u32 = .{ 6427, 2153, 5153, 2879 };
    const hadamard: zm.Mat = .{
        .{ 1, 1, 1, 1 },
        .{ 1, -1, 1, -1 },
        .{ 1, 1, -1, -1 },
        .{ 1, -1, -1, 1 },
    };
    const householder: zm.Mat = .{
        .{ 0.5, -0.5, -0.5, -0.5 },
        .{ -0.5, 0.5, -0.5, -0.5 },
        .{ -0.5, -0.5, 0.5, -0.5 },
        .{ -0.5, -0.5, -0.5, 0.5 },
    };

    diffuser_buffers: [4][4][4096]f32,
    diffuser_cursors: [4][4]u32,
    feedback_buffers: [4][8192]f32,
    feedback_cursors: [4]u32,
    feedback_filter_state: @Vector(4, f32),
    feedback_gain: f32 = 0.9,

    const init = std.mem.zeroInit(Reverb, .{});

    fn apply(rev: *Reverb, samples: [frame_size]f32, result: *[frame_size]f32) void {
        // split into channels
        var chunk: [4][frame_size]f32 = .{ samples, samples, samples, samples };
        for (0..frame_size) |i| {
            chunk[0][i] *= 0.25;
            chunk[1][i] *= 0.25;
            chunk[2][i] *= 0.25;
            chunk[3][i] *= 0.25;
        }
        // diffusion
        for (
            diffuser_delays,
            diffuser_shuffles,
            diffuser_polarities,
            0..,
        ) |delays, shuffles, polarities, i| {
            for (0..4) |k| {
                for (0..frame_size) |j| {
                    const sample = chunk[k][j];
                    const cursor = rev.diffuser_cursors[k][i];
                    chunk[k][j] = rev.diffuser_buffers[k][i][cursor];
                    rev.diffuser_buffers[k][i][cursor] = sample;
                    rev.diffuser_cursors[k][i] = (cursor + 1) % delays[k];
                }
            }
            for (0..frame_size) |j| {
                const s: [4]f32 = .{
                    chunk[shuffles[0]][j] * polarities[0],
                    chunk[shuffles[1]][j] * polarities[1],
                    chunk[shuffles[2]][j] * polarities[2],
                    chunk[shuffles[3]][j] * polarities[3],
                };
                chunk[0][j] = s[0];
                chunk[1][j] = s[1];
                chunk[2][j] = s[2];
                chunk[3][j] = s[3];
            }
            for (0..frame_size / 4) |j| {
                var a: zm.Mat = undefined;
                a[0] = chunk[0][j * 4 ..][0..4].*;
                a[1] = chunk[1][j * 4 ..][0..4].*;
                a[2] = chunk[2][j * 4 ..][0..4].*;
                a[3] = chunk[3][j * 4 ..][0..4].*;
                // a = zm.mul(a, hadamard);
                a = zm.mul(hadamard, a);
                chunk[0][j * 4 ..][0..4].* = a[0];
                chunk[1][j * 4 ..][0..4].* = a[1];
                chunk[2][j * 4 ..][0..4].* = a[2];
                chunk[3][j * 4 ..][0..4].* = a[3];
            }
        }
        // feedforward
        for (0..frame_size) |i| result[i] += chunk[0][i] + chunk[1][i] + chunk[2][i] + chunk[3][i];
        // feedback
        for (0..frame_size) |j| {
            const current: @Vector(4, f32) = .{
                chunk[0][j],
                chunk[1][j],
                chunk[2][j],
                chunk[3][j],
            };
            for (0..4) |i| {
                const cursor = rev.feedback_cursors[i];
                chunk[i][j] = rev.feedback_buffers[i][cursor];
            }
            var future: @Vector(4, f32) = .{
                chunk[0][j],
                chunk[1][j],
                chunk[2][j],
                chunk[3][j],
            };
            future *= @as(@Vector(4, f32), @splat(rev.feedback_gain));
            const alpha: @Vector(4, f32) = @splat(0.2);
            future = alpha * future + (zm.f32x4s(1.0) - alpha) * rev.feedback_filter_state;
            rev.feedback_filter_state = future;
            future = zm.mul(householder, future);
            future += current;
            for (0..4) |i| {
                const cursor = rev.feedback_cursors[i];
                rev.feedback_buffers[i][cursor] = future[i];
                rev.feedback_cursors[i] = (cursor + 1) % feedback_delays[i];
            }
        }
        // mix
        for (0..frame_size) |i| {
            result[i] += chunk[0][i];
            result[i] += chunk[1][i];
            result[i] += chunk[2][i];
            result[i] += chunk[3][i];
        }

        for (0..frame_size) |i| result[i] *= 0.5;
    }
};

const Sound = struct {
    buf: [*c]u8,
    len: u32,

    fn init(filename: [*c]const u8) !Sound {
        // seems like a sensible design would be to preconvert all input to a known format
        var raw_spec: sdl.c.SDL_AudioSpec = undefined;
        var buf: [*c]u8 = null;
        var len: u32 = 0;
        if (!sdl.c.SDL_LoadWAV(filename, &raw_spec, &buf, &len)) {
            log.err("SDL_LoadWAV: {s}", .{sdl.c.SDL_GetError()});
            return error.Sdl;
        }
        defer sdl.c.SDL_free(buf);
        var sound: Sound = .{
            .buf = null,
            .len = 0,
        };
        // weird mismatch between the u32 len from load and the c_int lens in convert
        var len2: c_int = 0;
        if (!sdl.c.SDL_ConvertAudioSamples(
            &raw_spec,
            buf,
            @intCast(len),
            &sound_effect_spec,
            &sound.buf,
            &len2,
        )) {
            log.err("SDL_ConvertAudioSamples: {s}", .{sdl.c.SDL_GetError()});
            return error.Sdl;
        }
        sound.len = @intCast(len2);
        return sound;
    }

    fn deinit(sound: *Sound) void {
        sdl.c.SDL_free(sound.buf);
        sound.* = undefined;
    }

    fn samples(sound: Sound) []f32 {
        std.debug.assert(@intFromPtr(sound.buf) % 4 == 0);
        std.debug.assert(sound.len % @sizeOf(f32) == 0);
        const p: [*]f32 = @alignCast(@ptrCast(sound.buf));
        return p[0 .. sound.len / @sizeOf(f32)];
    }
};

fn dopplerReflectionResample(
    p: *const Playing,
    prev_dist: f32,
    dist: f32,
    frame_index: usize,
    i: usize,
    total_frames: usize,
    samples: []const f32,
) f32 {
    const d = std.math.lerp(
        prev_dist,
        dist,
        @as(f32, @floatFromInt(128 * frame_index + i)) /
            @as(f32, @floatFromInt(128 * total_frames)),
    );
    // std.debug.print("{} {} {}\n", .{ frame_index, total_frames, d });
    const foff = @as(f32, @floatFromInt(p.cursor)) +
        4 * @as(f32, @floatFromInt(samples.len)) -
        44100 * d / speed_of_sound;
    const ioff = @as(usize, @intFromFloat(foff));
    const beta = foff - @trunc(foff);
    const sample = std.math.lerp(
        samples[(ioff + i) % samples.len],
        samples[(ioff + i - 1) % samples.len],
        beta,
    );
    return sample;
}
