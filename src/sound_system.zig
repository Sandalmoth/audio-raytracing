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
    elevation_resolution: f32,
    azimuth_resolution: f32,
    irs_l: []const []const []const f32,
    irs_r: []const []const []const f32,
} = @import("hrtf.zon");

const frame_size = 128;

gpa: std.mem.Allocator,
sounds: std.ArrayListUnmanaged(Sound),
playing: std.AutoArrayHashMapUnmanaged(usize, Playing),
playing_counter: usize,
stream: *sdl.c.SDL_AudioStream,
listener: zm.Vec,
stereo_frame_buffer: [2 * frame_size][2]f32,

pub fn init(gpa: std.mem.Allocator) !*SoundSystem {
    const system = try gpa.create(SoundSystem);
    errdefer gpa.destroy(system);

    system.gpa = gpa;
    system.sounds = .empty;
    system.playing = .empty;
    system.playing_counter = 0;
    system.listener = zm.f32x4s(0.0);
    system.stereo_frame_buffer = std.mem.zeroes([2 * frame_size][2]f32);
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
    var n_samples = @divTrunc(additional_amount, 2 * @sizeOf(f32));
    _ = total_amount;

    while (n_samples > 0) : (n_samples -= 128) {
        const ambisonic = system.buildAmbisonic();
        system.ambisonicToStereo(ambisonic);
        if (!sdl.c.SDL_PutAudioStreamData(
            stream,
            &system.stereo_frame_buffer[0],
            2 * frame_size * @sizeOf(f32),
        )) {
            log.err("SDL_PutAudioStreamData: {s}", .{sdl.c.SDL_GetError()});
        }

        for (0..frame_size) |i| {
            system.stereo_frame_buffer[i] = system.stereo_frame_buffer[2 * i];
            system.stereo_frame_buffer[2 * i] = .{ 0.0, 0.0 };
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
        const w: f32, const x: f32, const y: f32, const z: f32 = blk: {
            // if distance is below threshold, smoothly scale to 0 directional component
            const dir = system.listener - p.pos;
            const len = zm.length3(dir)[0];
            if (len < 1e-9) break :blk .{ std.math.sqrt1_2, 0.0, 0.0, 0.0 };
            const ndir = zm.normalize3(dir);
            const t = 0.1;
            if (len < t) {
                const d = ndir * zm.splat(zm.F32x4, len / t);
                break :blk .{ std.math.sqrt1_2, d[0], d[1], d[2] };
            } else {
                break :blk .{ std.math.sqrt1_2, ndir[0], ndir[1], ndir[2] };
            }
        };
        const s = system.sounds.items[p.sound];
        const samples = s.samples();
        if (p.repeat) {
            for (0..128) |i| {
                buf[0][i] += w * samples[(p.cursor + i) % samples.len];
                buf[1][i] += x * samples[(p.cursor + i) % samples.len];
                buf[2][i] += y * samples[(p.cursor + i) % samples.len];
                buf[3][i] += z * samples[(p.cursor + i) % samples.len];
            }
            p.cursor += 128;
        } else {
            const end = @min(p.cursor + 128, samples.len);
            for (p.cursor..end) |i| {
                buf[0][i - p.cursor] += w * samples[i];
                buf[1][i - p.cursor] += x * samples[i];
                buf[2][i - p.cursor] += y * samples[i];
                buf[3][i - p.cursor] += z * samples[i];
            }
            p.cursor = end;
            if (p.cursor == samples.len) p.finished = true;
        }
    }
    return buf;
}

fn ambisonicToStereo(system: *SoundSystem, ambisonic: [4][frame_size]f32) void {
    const scale = 1.0 / 14.0;
    var conv_bufs: [2][2 * frame_size]f32 = undefined;
    for ([_][2]f32{
        .{ 0.0 * std.math.pi, 0.0 },
        .{ 0.5 * std.math.pi, 0.0 },
        .{ 1.0 * std.math.pi, 0.0 },
        .{ 1.5 * std.math.pi, 0.0 },
        .{ 0.0, -0.5 * std.math.pi },
        .{ 0.0, 0.5 * std.math.pi },
        .{ 0.25 * std.math.pi, 0.25 * std.math.pi },
        .{ 0.75 * std.math.pi, 0.25 * std.math.pi },
        .{ 1.25 * std.math.pi, 0.25 * std.math.pi },
        .{ 1.75 * std.math.pi, 0.25 * std.math.pi },
        .{ 0.25 * std.math.pi, -0.25 * std.math.pi },
        .{ 0.75 * std.math.pi, -0.25 * std.math.pi },
        .{ 1.25 * std.math.pi, -0.25 * std.math.pi },
        .{ 1.75 * std.math.pi, -0.25 * std.math.pi },
    }) |ae| {
        const azim, const elev = ae;
        const irs = getHrtfIr(azim, elev);
        convolve(&ambisonic[0], irs[0], &conv_bufs[0]);
        convolve(&ambisonic[0], irs[1], &conv_bufs[1]);

        for (0..frame_size) |i| {
            system.stereo_frame_buffer[i][0] += conv_bufs[0][i] * scale;
            system.stereo_frame_buffer[i][1] += conv_bufs[1][i] * scale;
            system.stereo_frame_buffer[i + frame_size][0] = conv_bufs[0][i + frame_size] * scale;
            system.stereo_frame_buffer[i + frame_size][1] = conv_bufs[1][i + frame_size] * scale;
        }
    }

    // for (&ambisonic[0], 0..) |sample, i| {
    //     system.stereo_frame_buffer[i][0] = sample;
    //     system.stereo_frame_buffer[i][1] = sample;
    // }
}

// class HRTFStreamProcessor:
//     def __init__(self, hrtf_data, frame_size=256):
//         self.hrtf_data = hrtf_data
//         self.frame_size = frame_size
//         self.input_buffer = np.zeros(frame_size * 2)  # Buffer for overlap-add
//         self.output_buffer = np.zeros(frame_size * 2)  # Buffer for overlap-add

//     def process_frame(self, mono_frame, azimuth, elevation):
//         # Apply distance attenuation (if needed)
//         # mono_frame = apply_distance_attenuation(mono_frame, distance)

//         # Get the HRTF for the current direction
//         hrtf_left, hrtf_right = interpolate_hrtf(azimuth, elevation, self.hrtf_data)

//         # Convolve the frame with the HRTF using overlap-add
//         padded_frame = np.concatenate([mono_frame, np.zeros(len(hrtf_left) - 1)])
//         spatialized_left = convolve(padded_frame, hrtf_left, mode='valid')
//         spatialized_right = convolve(padded_frame, hrtf_right, mode='valid')

//         # Overlap-add to handle streaming
//         self.output_buffer[:self.frame_size] += spatialized_left[:self.frame_size]
//         self.output_buffer[self.frame_size:] = spatialized_left[self.frame_size:]

//         # Output the current frame
//         output_frame = self.output_buffer[:self.frame_size]
//         self.output_buffer = np.roll(self.output_buffer, -self.frame_size)
//         self.output_buffer[-self.frame_size:] = 0  # Clear the end for next frame

//         return output_frame

fn getHrtfIr(azimuth: f32, elevation: f32) [2][]const f32 {
    std.debug.assert(azimuth >= 0);
    std.debug.assert(azimuth <= std.math.tau);
    std.debug.assert(elevation >= -0.5 * std.math.pi);
    std.debug.assert(elevation <= 0.5 * std.math.pi);

    const ix_azim = @as(usize, @intFromFloat(@round(
        azimuth / std.math.tau * @as(f32, @floatFromInt(hrtf.irs_l[0].len)),
    ))) % hrtf.irs_l[0].len;
    const ix_elev = @as(usize, @intFromFloat(@round(
        (elevation + 0.5 * std.math.pi) / std.math.pi * @as(f32, @floatFromInt(hrtf.irs_l.len)),
    ))) % hrtf.irs_l.len;

    return .{
        hrtf.irs_l[ix_elev][ix_azim],
        hrtf.irs_r[ix_elev][ix_azim],
    };
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
    cursor: usize = 0,
    repeat: bool = false,
    finished: bool = false,
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
