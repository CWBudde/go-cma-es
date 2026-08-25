"use strict";

window.Render = (function () {
  const palette = {
    low: [10, 18, 35],
    mid: [32, 61, 86],
    high: [206, 120, 61],
    sample: "rgba(227, 239, 242, .78)",
    mean: "#ffca58",
    ellipse: "#50dfc5",
    best: "#ff6f91",
    optimum: "#eef7f5",
    baseline: "#c38cff",
    grid: "rgba(223, 238, 240, .12)",
    dim: "#81949b",
    text: "#e3eef0",
  };

  function mix(a, b, t) {
    return a.map((value, index) => Math.round(value + (b[index] - value) * t));
  }

  function ramp(t) {
    const color = t < 0.62 ? mix(palette.low, palette.mid, t / 0.62) : mix(palette.mid, palette.high, (t - 0.62) / 0.38);
    return color;
  }

  function backdrop(landscape) {
    const canvas = document.createElement("canvas");
    canvas.width = landscape.width;
    canvas.height = landscape.height;
    const context = canvas.getContext("2d");
    const image = context.createImageData(canvas.width, canvas.height);

    for (let index = 0; index < landscape.values.length; index += 1) {
      const color = ramp(Math.pow(landscape.values[index], 0.72));
      const offset = index * 4;
      image.data[offset] = color[0];
      image.data[offset + 1] = color[1];
      image.data[offset + 2] = color[2];
      image.data[offset + 3] = 255;
    }
    context.putImageData(image, 0, 0);
    return canvas;
  }

  function point(canvas, landscape, x, y) {
    const span = landscape.upper - landscape.lower;
    return [
      ((x - landscape.lower) / span) * canvas.width,
      ((landscape.upper - y) / span) * canvas.height,
    ];
  }

  function base(context, canvas, landscape, image) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.imageSmoothingEnabled = true;
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    context.strokeStyle = palette.grid;
    context.lineWidth = 1;
    const zero = point(canvas, landscape, 0, 0);
    context.beginPath();
    context.moveTo(zero[0], 0);
    context.lineTo(zero[0], canvas.height);
    context.moveTo(0, zero[1]);
    context.lineTo(canvas.width, zero[1]);
    context.stroke();

    for (const minimum of landscape.minima || []) drawCross(context, ...point(canvas, landscape, minimum[0], minimum[1]), palette.optimum, 7);
  }

  function drawCross(context, x, y, color, size) {
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(x - size, y - size);
    context.lineTo(x + size, y + size);
    context.moveTo(x + size, y - size);
    context.lineTo(x - size, y + size);
    context.stroke();
  }

  function trail(context, canvas, landscape, values, frame, color) {
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    for (let index = 0; index <= frame; index += 1) {
      const p = point(canvas, landscape, values[index * 2], values[index * 2 + 1]);
      if (index === 0) context.moveTo(...p);
      else context.lineTo(...p);
    }
    context.stroke();
  }

  function population(context, canvas, landscape, values, frame, lambda, color) {
    context.fillStyle = color;
    for (let index = 0; index < lambda; index += 1) {
      const offset = (frame * lambda + index) * 2;
      const p = point(canvas, landscape, values[offset], values[offset + 1]);
      context.beginPath();
      context.arc(p[0], p[1], index < lambda / 2 ? 3.2 : 2.2, 0, Math.PI * 2);
      context.fill();
    }
  }

  function distribution(context, canvas, landscape, run, frame) {
    const meanOffset = frame * 2;
    const matrixOffset = frame * 4;
    const mx = run.mean[meanOffset];
    const my = run.mean[meanOffset + 1];
    const matrix = run.ellipse.subarray(matrixOffset, matrixOffset + 4);
    const span = landscape.upper - landscape.lower;
    const scale = canvas.width / span;
    const center = point(canvas, landscape, mx, my);

    context.strokeStyle = palette.ellipse;
    context.lineWidth = 2.5;
    context.beginPath();
    for (let sample = 0; sample <= 80; sample += 1) {
      const angle = (sample / 80) * Math.PI * 2;
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      const x = center[0] + scale * (matrix[0] * cosine + matrix[1] * sine);
      const y = center[1] - scale * (matrix[2] * cosine + matrix[3] * sine);
      if (sample === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();
    context.fillStyle = palette.mean;
    context.beginPath();
    context.arc(center[0], center[1], 5, 0, Math.PI * 2);
    context.fill();
  }

  function run(canvas, landscape, image, data, frame, options) {
    const context = canvas.getContext("2d");
    base(context, canvas, landscape, image);
    if (data.bestTrail) trail(context, canvas, landscape, data.bestTrail, frame, (options && options.trail) || palette.best);
    population(context, canvas, landscape, data.population, frame, data.lambda, (options && options.samples) || palette.sample);
    if (data.ellipse && data.mean) distribution(context, canvas, landscape, data, frame);
    else if (data.mean) {
      const offset = frame * 2;
      const p = point(canvas, landscape, data.mean[offset], data.mean[offset + 1]);
      context.fillStyle = (options && options.mean) || palette.baseline;
      context.beginPath();
      context.arc(p[0], p[1], 5, 0, Math.PI * 2);
      context.fill();
    }
  }

  function chart(canvas, series, frame, logScale) {
    const context = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    const pad = { left: 58, right: 16, top: 18, bottom: 30 };
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#0d151c";
    context.fillRect(0, 0, width, height);

    const transformed = [];
    for (const line of series) {
      for (let index = 0; index <= Math.min(frame, line.values.length - 1); index += 1) {
        const raw = Math.max(Number.MIN_VALUE, line.values[index]);
        transformed.push(logScale ? Math.log10(raw) : raw);
      }
    }
    if (!transformed.length) return;
    let low = Math.min(...transformed);
    let high = Math.max(...transformed);
    if (low === high) high = low + 1;

    context.strokeStyle = palette.grid;
    context.fillStyle = palette.dim;
    context.font = "12px ui-monospace, monospace";
    context.beginPath();
    for (let row = 0; row <= 4; row += 1) {
      const y = pad.top + ((height - pad.top - pad.bottom) * row) / 4;
      context.moveTo(pad.left, y);
      context.lineTo(width - pad.right, y);
      const value = high - ((high - low) * row) / 4;
      context.fillText(logScale ? `10^${value.toFixed(1)}` : value.toPrecision(3), 4, y + 4);
    }
    context.stroke();

    for (const line of series) {
      context.strokeStyle = line.color;
      context.lineWidth = 2;
      context.beginPath();
      const end = Math.min(frame, line.values.length - 1);
      for (let index = 0; index <= end; index += 1) {
        const raw = Math.max(Number.MIN_VALUE, line.values[index]);
        const value = logScale ? Math.log10(raw) : raw;
        const x = pad.left + ((width - pad.left - pad.right) * index) / Math.max(1, line.values.length - 1);
        const y = pad.top + ((height - pad.top - pad.bottom) * (high - value)) / (high - low);
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
    }
  }

  function restarts(canvas, landscape, image, markers) {
    const context = canvas.getContext("2d");
    base(context, canvas, landscape, image);
    context.font = "bold 13px ui-monospace, monospace";
    for (let index = 0; index < markers.length / 4; index += 1) {
      const offset = index * 4;
      const best = point(canvas, landscape, markers[offset], markers[offset + 1]);
      const basin = point(canvas, landscape, markers[offset + 2], markers[offset + 3]);
      context.strokeStyle = palette.ellipse;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(...best);
      context.lineTo(...basin);
      context.stroke();
      context.fillStyle = palette.mean;
      context.beginPath();
      context.arc(best[0], best[1], 5, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = palette.text;
      context.fillText(String(index + 1), best[0] + 8, best[1] - 8);
      context.strokeStyle = palette.baseline;
      context.strokeRect(basin[0] - 5, basin[1] - 5, 10, 10);
    }
  }

  return { backdrop, run, chart, restarts, palette };
})();
