//go:build js && wasm

package main

import (
	"syscall/js"
	"unsafe"
)

type float32Sink struct {
	f32      js.Value
	u8       js.Value
	capacity int
}

func newFloat32Sink(length int) float32Sink {
	length = max(0, length)
	buffer := js.Global().Get("ArrayBuffer").New(length * 4)

	return float32Sink{
		f32:      js.Global().Get("Float32Array").New(buffer),
		u8:       js.Global().Get("Uint8Array").New(buffer),
		capacity: length,
	}
}

func sinkFor(out js.Value, key string, length int) float32Sink {
	if isObject(out) {
		candidate := out.Get(key)
		if isObject(candidate) {
			f32, u8 := candidate.Get("f32"), candidate.Get("u8")
			if isObject(f32) && isObject(u8) && f32.Length() >= length {
				return float32Sink{f32: f32, u8: u8, capacity: f32.Length()}
			}
		}
	}

	return newFloat32Sink(length)
}

func (sink float32Sink) write(data []float32) js.Value {
	if len(data) > 0 {
		js.CopyBytesToJS(sink.u8, float32Bytes(data))
	}

	if sink.capacity == len(data) {
		return sink.f32
	}

	return sink.f32.Call("subarray", 0, len(data))
}

func float32Bytes(data []float32) []byte {
	if len(data) == 0 {
		return nil
	}

	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
}

func putFloats(result map[string]any, out js.Value, key string, data []float32) {
	result[key] = sinkFor(out, key, len(data)).write(data)
}

func narrow(values []float64) []float32 {
	result := make([]float32, len(values))
	for index, value := range values {
		result[index] = float32(value)
	}

	return result
}
