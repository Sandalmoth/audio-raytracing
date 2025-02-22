const std = @import("std");

const math = @import("math.zig");
const sdl = @import("sdl.zig");

const Input = @This();

const ButtonState = struct {
    held: bool = false,
    pressed: bool = false,
    released: bool = false,
    mouse_pos_pressed: math.Vec2f = .zeros,
    mouse_pos_released: math.Vec2f = .zeros,
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
};

map: std.AutoArrayHashMap(SdlInput, GameInput),
mouse_pos: math.Vec2f,
mouse_delta: math.Vec2f,
states: std.EnumArray(GameInput, ButtonState),

pub fn init(alloc: std.mem.Allocator) Input {
    return .{
        .map = std.AutoArrayHashMap(SdlInput, GameInput).init(alloc),
        .mouse_pos = .zeros,
        .mouse_delta = .zeros,
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
            input.mouse_pos = .{ .x = event.motion.x, .y = event.motion.y };
            input.mouse_delta = math.add(
                input.mouse_delta,
                math.Vec2f{ .x = event.motion.xrel, .y = event.motion.yrel },
            );
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
    input.mouse_delta = .{ .x = 0, .y = 0 };
    var it = input.states.iterator();
    while (it.next()) |kv| {
        kv.value.pressed = false;
        kv.value.released = false;
    }
}
