const std = @import("std");

const sdl = @import("sdl.zig");

const log = std.log;

pub const ticks_per_second = 43;
pub const tick: f32 = 1.0 / @as(f32, @floatFromInt(ticks_per_second));
pub const tick_ns: u64 = 1000_000_000 / ticks_per_second;
pub const max_tick_ns: u64 = @intFromFloat(0.1 * 1e9);

pub fn main() !void {
    sdl.c.SDL_SetMainReady();

    if (!sdl.c.SDL_Init(sdl.c.SDL_INIT_VIDEO)) {
        log.err("SDL_Init: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    }
    defer sdl.c.SDL_Quit();

    const window: *sdl.c.SDL_Window, const renderer: *sdl.c.SDL_Renderer = blk: {
        var window: ?*sdl.c.SDL_Window = null;
        var renderer: ?*sdl.c.SDL_Renderer = null;
        if (!sdl.c.SDL_CreateWindowAndRenderer("audio-raytracing", 640, 360, 0, &window, &renderer)) {
            log.err("SDL_CreateWindowAndRenderer: {s}", .{sdl.c.SDL_GetError()});
            return error.Sdl;
        }
        break :blk .{ window.?, renderer.? };
    };
    defer sdl.c.SDL_DestroyWindow(window);
    defer sdl.c.SDL_DestroyRenderer(renderer);

    if (!sdl.c.SDL_SetRenderVSync(renderer, 1)) {
        log.err("SDL_SetRenderVSync: {s}", .{sdl.c.SDL_GetError()});
        return error.Sdl;
    }

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
        }

        while (lag >= tick_ns) {
            // update
            lag -= tick_ns;
        }

        _ = sdl.c.SDL_SetRenderDrawColor(renderer, 0x18, 0x18, 0x18, 0xff);
        _ = sdl.c.SDL_RenderClear(renderer);

        const alpha = @as(f32, @floatFromInt(lag)) / @as(f32, @floatFromInt(tick_ns));
        _ = alpha;

        _ = sdl.c.SDL_RenderPresent(renderer);
    }
}
