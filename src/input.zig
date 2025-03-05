const std = @import("std");

const sdl = @import("sdl.zig");
const zm = @import("zmath");

const Input = @This();

const ButtonState = struct {
    held: bool = false,
    pressed: bool = false,
    released: bool = false,
    mouse_pos_pressed: zm.Vec = zm.f32x4s(0),
    mouse_pos_released: zm.Vec = zm.f32x4s(0),
};

const SdlInput = union(enum) {
    mouse: sdl.c.Uint8,
    keyboard: sdl.c.SDL_Scancode,
};

const GameInput = enum {
    fire,
    grab,
    interact,
    forward,
    left,
    right,
    backward,
    up,
    down,
};

map: std.AutoArrayHashMap(SdlInput, GameInput),
mouse_pos: zm.Vec,
mouse_delta: zm.Vec,
states: std.EnumArray(GameInput, ButtonState),

pub fn init(alloc: std.mem.Allocator) Input {
    return .{
        .map = std.AutoArrayHashMap(SdlInput, GameInput).init(alloc),
        .mouse_pos = zm.f32x4s(0),
        .mouse_delta = zm.f32x4s(0),
        .states = std.EnumArray(GameInput, ButtonState).initFill(.{}),
    };
}

pub fn deinit(input: *Input) void {
    input.map.deinit();
    input.* = undefined;
}

pub fn peek(input: Input, game_input: GameInput) ButtonState {
    return input.states.get(game_input);
}

pub fn consume(input: *Input, game_input: GameInput) ButtonState {
    const result = input.states.get(game_input);
    const state = input.states.getPtr(game_input);
    state.held = false;
    state.pressed = false;
    state.released = false;
    return result;
}

pub fn accumulate(input: *Input, event: sdl.c.SDL_Event) void {
    dispatch: switch (event.type) {
        sdl.c.SDL_EVENT_MOUSE_MOTION => {
            input.mouse_pos = .{ event.motion.x, event.motion.y, 0.0, 0.0 };
            input.mouse_delta += zm.f32x4(event.motion.xrel, event.motion.yrel, 0.0, 0.0);
        },
        sdl.c.SDL_EVENT_MOUSE_BUTTON_DOWN => {
            const game_input = input.map.get(
                .{ .mouse = event.button.button },
            ) orelse break :dispatch;
            const state = input.states.getPtr(game_input);
            state.held = true;
            if (!state.pressed) state.mouse_pos_pressed = input.mouse_pos;
            state.pressed = true;
        },
        sdl.c.SDL_EVENT_MOUSE_BUTTON_UP => {
            const game_input = input.map.get(
                .{ .mouse = event.button.button },
            ) orelse break :dispatch;
            const state = input.states.getPtr(game_input);
            state.held = false;
            state.mouse_pos_released = input.mouse_pos;
            state.released = true;
        },
        sdl.c.SDL_EVENT_KEY_DOWN => {
            const game_input = input.map.get(
                .{ .keyboard = event.key.scancode },
            ) orelse break :dispatch;
            const state = input.states.getPtr(game_input);
            state.held = true;
            if (!state.pressed) state.mouse_pos_pressed = input.mouse_pos;
            state.pressed = true;
        },
        sdl.c.SDL_EVENT_KEY_UP => {
            const game_input = input.map.get(
                .{ .keyboard = event.key.scancode },
            ) orelse break :dispatch;
            const state = input.states.getPtr(game_input);
            state.held = false;
            state.mouse_pos_released = input.mouse_pos;
            state.released = true;
        },
        else => {},
    }
}

pub fn decay(input: *Input) void {
    input.mouse_delta = zm.f32x4s(0);
    var it = input.states.iterator();
    while (it.next()) |kv| {
        kv.value.pressed = false;
        kv.value.released = false;
    }
}
