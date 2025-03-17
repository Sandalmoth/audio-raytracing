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
    // var t = std.time.Timer.start() catch unreachable;
    // defer std.debug.print("{d:.2}\n", .{@as(f64, @floatFromInt(t.lap())) * 1e-6});
    const system = @as(?*SoundSystem, @alignCast(@ptrCast(ctx))).?;
    system.mutex.lock();
    defer system.mutex.unlock();

    var n_samples = @divTrunc(additional_amount, 2 * @sizeOf(f32));
    _ = total_amount;

    while (n_samples > 0) : (n_samples -= 128) {
        var ambisonic = system.buildAmbisonic();
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

fn buildAmbisonic(system: *SoundSystem) [4][frame_size]f32 {
    var buf = std.mem.zeroes([4][frame_size]f32);
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

        if (p.repeat) {
            for (0..128) |i| {
                for (0..4) |j| {
                    const sample = p.attenuation_eq.apply(
                        samples[(p.cursor + i) % samples.len],
                    );
                    buf[j][i] += sh[j] * sample * p.gain;
                }
            }
            p.cursor += 128;
        } else {
            const end = @min(p.cursor + 128, samples.len);
            for (p.cursor..end) |i| {
                for (0..4) |j| {
                    const sample = p.attenuation_eq.apply(
                        samples[i],
                    );
                    buf[j][i - p.cursor] += sh[j] * sample * p.gain;
                }
            }
            p.cursor = end;
            if (p.cursor == samples.len) p.finished = true;
        }
    }
    return buf;
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

const Playing = struct {
    sound: usize, // should be a type safe id in a proper implementation
    pos: zm.Vec,
    gain: f32 = 1.0,
    cursor: usize = 0,
    repeat: bool = false,
    finished: bool = false,
    attenuation_eq: Equalizer = .{},
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
