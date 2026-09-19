/** Black-and-white canvas line charts. */
(function (global) {
  function cssSize(canvas) {
    // Prefer laid-out CSS size. Never use canvas.width (backing-store pixels) as
    // logical width — that pushes strokes far outside the visible box after DPR scale.
    const rect = canvas.getBoundingClientRect();
    let w = rect.width;
    let h = rect.height;
    if (!w || !h) {
      w = canvas.clientWidth || 400;
      h = canvas.clientHeight || 160;
    }
    if (!w) w = 400;
    if (!h) h = 160;
    return { w: w, h: h };
  }

  function clear(canvas) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const size = cssSize(canvas);
    const w = size.w;
    const h = size.h;
    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "#000";
    ctx.fillStyle = "#000";
    ctx.lineWidth = 1.25;
    return { ctx: ctx, w: w, h: h };
  }

  function drawFrame(ctx, w, h, pad) {
    ctx.save();
    ctx.strokeStyle = "#e5e5e5";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.l + 0.5, pad.t + 0.5, w - pad.l - pad.r - 1, h - pad.t - pad.b - 1);
    ctx.restore();
  }

  function seriesExtents(seriesList) {
    let min = Infinity;
    let max = -Infinity;
    let n = 0;
    for (let s = 0; s < seriesList.length; s++) {
      const values = seriesList[s].values || [];
      n = Math.max(n, values.length);
      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (Number.isFinite(v)) {
          min = Math.min(min, v);
          max = Math.max(max, v);
        }
      }
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      min = 0;
      max = 1;
    }
    if (min === max) {
      max = min + 1;
    }
    const span = max - min;
    min -= span * 0.08;
    max += span * 0.08;
    return { min: min, max: max, n: n };
  }

  function plot(canvas, seriesList) {
    if (!canvas) return;
    const cleared = clear(canvas);
    const ctx = cleared.ctx;
    const w = cleared.w;
    const h = cleared.h;
    const pad = { t: 8, r: 10, b: 18, l: 36 };
    drawFrame(ctx, w, h, pad);
    const extents = seriesExtents(seriesList || []);
    const min = extents.min;
    const max = extents.max;
    const n = extents.n;
    if (n < 1) return;

    const plotW = w - pad.l - pad.r;
    const plotH = h - pad.t - pad.b;

    ctx.save();
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#525252";
    ctx.textAlign = "right";
    ctx.fillText(max.toFixed(1), pad.l - 4, pad.t + 8);
    ctx.fillText(min.toFixed(1), pad.l - 4, pad.t + plotH);
    ctx.restore();

    (seriesList || []).forEach(function (series, idx) {
      const values = series.values || [];
      if (!values.length) return;

      ctx.beginPath();
      ctx.strokeStyle = "#000";
      ctx.fillStyle = "#000";
      ctx.lineWidth = idx === 0 ? 1.75 : 1.25;
      if (idx > 0) ctx.setLineDash([4, 3]);

      if (n === 1) {
        // Single Step → one sample: draw a short bar so it isn't just a lonely dot.
        const v = values[0];
        if (Number.isFinite(v)) {
          const x = pad.l + plotW / 2;
          const y = pad.t + plotH - ((v - min) / (max - min)) * plotH;
          const half = Math.min(28, plotW * 0.2);
          ctx.moveTo(x - half, y);
          ctx.lineTo(x + half, y);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.beginPath();
          ctx.arc(x, y, 3.5, 0, Math.PI * 2);
          ctx.fill();
        }
        return;
      }

      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (!Number.isFinite(v)) continue;
        const x = pad.l + (i / (n - 1)) * plotW;
        const y = pad.t + plotH - ((v - min) / (max - min)) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    });
  }

  global.AeroCharts = { plot: plot };
})(window);
