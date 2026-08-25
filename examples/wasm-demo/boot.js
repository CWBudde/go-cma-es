"use strict";

window.Demo = (function () {
  const $ = (id) => document.getElementById(id);

  function setStatus(message, state) {
    const node = $("status");
    if (!node) return;
    node.textContent = message;
    node.dataset.state = state || "";
  }

  function call(name, options) {
    if (!window.cmaes || typeof window.cmaes[name] !== "function") {
      setStatus(`the wasm module does not export ${name}()`, "error");
      return null;
    }

    try {
      const result = window.cmaes[name](options || {});
      if (!result || result.error) {
        setStatus((result && result.error) || `${name} returned nothing`, "error");
        return null;
      }
      return result;
    } catch (error) {
      setStatus(`${name}: ${(error && error.message) || error}`, "error");
      return null;
    }
  }

  function cacheSinks(sinks, result, keys) {
    for (const key of keys) {
      const view = result[key];
      if (!view || !view.buffer) continue;
      sinks[key] = {
        f32: new Float32Array(view.buffer),
        u8: new Uint8Array(view.buffer),
      };
    }
  }

  function Transport(onFrame) {
    const play = $("play");
    const step = $("step");
    const scrub = $("scrub");
    const speed = $("speed");
    const readout = $("frameReadout");
    let frames = 0;
    let frame = 0;
    let playing = false;
    let last = 0;

    function render() {
      if (scrub) scrub.value = String(frame);
      if (readout) {
        readout.textContent = frames ? `generation ${frame + 1} / ${frames}` : "—";
      }
      onFrame(frame);
    }

    function setFrame(value) {
      frame = Math.max(0, Math.min(Number(value), Math.max(0, frames - 1)));
      render();
    }

    function setPlaying(value) {
      playing = value;
      if (play) {
        play.textContent = value ? "Pause" : "Play";
        play.setAttribute("aria-pressed", String(value));
      }
      if (!value) return;
      if (frame >= frames - 1) setFrame(0);
      last = 0;
      requestAnimationFrame(tick);
    }

    function tick(now) {
      if (!playing) return;
      const interval = 1000 / (24 * Number((speed && speed.value) || 1));
      if (now - last >= interval) {
        last = now;
        if (frame >= frames - 1) {
          setPlaying(false);
          return;
        }
        setFrame(frame + 1);
      }
      requestAnimationFrame(tick);
    }

    if (play) play.addEventListener("click", () => setPlaying(!playing));
    if (step) {
      step.addEventListener("click", () => {
        setPlaying(false);
        setFrame(frame + 1);
      });
    }
    if (scrub) {
      scrub.addEventListener("input", () => {
        setPlaying(false);
        setFrame(scrub.value);
      });
    }

    return {
      // reset rewinds to the first generation of a freshly computed run.
      // autoplay starts the replay immediately, which is what the Run buttons
      // want; a single-frame run stays paused because there is nothing to play.
      reset(count, autoplay) {
        frames = count;
        if (scrub) scrub.max = String(Math.max(0, frames - 1));
        setPlaying(false);
        setFrame(0);
        if (autoplay && frames > 1) setPlaying(true);
      },
      stop() {
        setPlaying(false);
      },
    };
  }

  async function load() {
    setStatus("Loading WebAssembly…", "loading");
    if (!WebAssembly.instantiateStreaming) {
      WebAssembly.instantiateStreaming = async (response, imports) =>
        WebAssembly.instantiate(await (await response).arrayBuffer(), imports);
    }

    const go = new Go();
    const response = await fetch("cmaes.wasm");
    if (!response.ok) throw new Error(`fetch cmaes.wasm: ${response.status}`);
    const module = await WebAssembly.instantiateStreaming(response, go.importObject);
    go.run(module.instance);
    await new Promise((resolve) => setTimeout(resolve, 0));

    const info = call("info", {});
    if (!info) throw new Error("the wasm module did not publish its capability table");
    const build = $("buildInfo");
    if (build) build.textContent = `v${info.version} · ${info.goVersion} · ${info.goos}/${info.goarch}`;
    return info;
  }

  function start(main) {
    load()
      .then(main)
      .catch((error) => setStatus((error && error.message) || error, "error"));
  }

  return { $, setStatus, call, cacheSinks, Transport, start };
})();
