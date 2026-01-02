import os
from jetengine_ext.llm import LLM
from jetengine_ext.sampling_params import SamplingParams
from transformers import AutoTokenizer

model_path = "/abs/path/of/your/model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# Initialize the LLM
llm = LLM(
    model_path,
    enforce_eager=True,
    tensor_parallel_size=1,
    mask_token_id=151669,   
    block_length=4
)

# Set sampling/generation parameters
sampling_params = SamplingParams(
    temperature=1.0,
    topk=0,
    topp=1.0,
    max_tokens=256,
    remasking_strategy="low_confidence_dynamic",
    block_length=4,
    denoising_steps=4,
    dynamic_threshold=0.9
)

# Prepare a simple chat-style prompt
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain what reinforcement learning is in simple terms."}],
    tokenize=False,
    add_generation_prompt=True
)

# Generate text
outputs = llm.generate_streaming([prompt], sampling_params)

output_text = outputs[0]["text"]
step_map = outputs[0]["first_unmask_times"]
token_ids   = outputs[0]["token_ids"]

#print(output_text)



















### output trace viewer
pieces = []
prev = ""
for i in range(len(token_ids)):
    cur = tokenizer.decode(token_ids[:i+1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    piece = cur[len(prev):]
    pieces.append(piece)
    prev = cur

# 2) 标记特殊 token（可在前端开关隐藏）
try:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
except Exception:
    special_ids = set()
is_special = [tid in special_ids for tid in token_ids]

# 3) 写出自包含 HTML 文件
import json, os

data = {
    "pieces": pieces,
    "step_map": step_map,
    "is_special": is_special,
}

# --- build data for HTML ---
step_map  = outputs[0].get("first_unmask_time") or outputs[0].get("first_unmask_times")
token_ids = outputs[0]["token_ids"]

# 逐 token 的增量片段，确保和 tokenizer 解码一致
pieces = []
prev = ""
for i in range(len(token_ids)):
    cur = tokenizer.decode(token_ids[:i+1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    piece = cur[len(prev):]
    pieces.append(piece)
    prev = cur

# 标记特殊 token（用于可选隐藏）
try:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
except Exception:
    special_ids = set()
is_special = [tid in special_ids for tid in token_ids]

import json, os

data = {
    "pieces": pieces,
    "step_map": step_map,
    "is_special": is_special,
}

# --- self-contained HTML (fixed-width |<MASK>| for masked tokens) ---
html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trace Viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{ --fg:#111; --muted:#888; --new:#0a7f2e; --mask:#aaa; --border:#ddd; }}
  body {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin:16px; color:var(--fg); }}
  h1 {{ font-size:18px; margin:0 0 12px 0; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:12px; }}
  .controls > * {{ font-size:14px; }}
  #viewer {{ border:1px solid var(--border); padding:12px; height:60vh; overflow:auto;
            white-space:pre-wrap; overflow-wrap:anywhere; word-break:normal; }}
  .masked {{ color:transparent; text-shadow:0 0 0 var(--mask); }}
  .unmasked {{ color:inherit; }}
  .new {{ background:#e9f7ee; outline:1px dashed #b7e1c3; }}
  .special {{ color:var(--muted); }}
  .meta {{ color:var(--muted); font-size:12px; margin-bottom:6px; }}
  .row {{ display:flex; gap:8px; align-items:center; }}
  input[type="range"] {{ width:300px; }}
  button {{ padding:4px 10px; }}
  select {{ padding:2px 6px; }}
  label {{ user-select:none; }}
  .maskToken {{ display:inline-block; width:7ch; text-align:center; white-space:nowrap; word-break:keep-all; }}
</style>
</head>
<body>
  <h1>Unmask Viewer</h1>
  <div class="meta" id="meta"></div>
  <div class="controls">
    <div class="row">
      <button id="play">Play</button>
      <button id="prev">-1</button>
      <input id="step" type="range" min="0" max="0" value="0">
      <button id="next">+1</button>
    </div>
    <div class="row">
      <label>Step: <span id="stepNum">0</span>/<span id="stepMax">0</span></label>
      <label style="margin-left:12px;">Speed:
        <select id="speed">
          <option value="1200">0.8×</option>
          <option value="800" selected>1×</option>
          <option value="500">1.6×</option>
          <option value="300">2.6×</option>
        </select>
      </label>
      <label style="margin-left:12px;">
        <input id="hideSpecial" type="checkbox"> Hide special tokens
      </label>
      <label style="margin-left:12px;">
        <input id="showMaskedShapes" type="checkbox" checked> Show masked placeholders
      </label>
    </div>
  </div>
  <div id="viewer"></div>

  <script id="data" type="application/json">{json.dumps(data, ensure_ascii=False)}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('data').textContent);
    const pieces = DATA.pieces;
    const steps  = DATA.step_map;
    const isSpec = DATA.is_special;

    const maxStep = Math.max(...steps);
    const viewer = document.getElementById('viewer');
    const stepInput = document.getElementById('step');
    const stepNum = document.getElementById('stepNum');
    const stepMax = document.getElementById('stepMax');
    const playBtn = document.getElementById('play');
    const prevBtn = document.getElementById('prev');
    const nextBtn = document.getElementById('next');
    const speedSel = document.getElementById('speed');
    const hideSpecial = document.getElementById('hideSpecial');
    const showMaskedShapes = document.getElementById('showMaskedShapes');
    const meta = document.getElementById('meta');

    stepInput.max = String(maxStep);
    stepMax.textContent = String(maxStep);

    function render(t) {{
      stepNum.textContent = String(t);
      const frag = document.createDocumentFragment();
      let revealed = 0, newly = 0;

      for (let i = 0; i < pieces.length; i++) {{
        if (hideSpecial.checked && isSpec[i]) continue;

        const span = document.createElement('span');
        const piece = pieces[i];

        if (steps[i] <= t) {{
          span.className = 'unmasked' + (steps[i] === t ? ' new' : '');
          span.textContent = piece;
          if (steps[i] === t) newly++;
          revealed++;
        }} else {{
          span.className = 'masked maskToken' + (isSpec[i] ? ' special' : '');
          span.textContent = showMaskedShapes.checked ? '|<MASK>|' : '';
        }}

        if (isSpec[i]) span.classList.add('special');
        frag.appendChild(span);
      }}

      viewer.innerHTML = '';
      viewer.appendChild(frag);
      meta.textContent = `Tokens revealed: ${{revealed}} / ${{pieces.length}}  |  Newly at step ${{t}}: ${{newly}}`;
    }}

    let timer = null;
    function play() {{
      if (timer) return;
      playBtn.textContent = 'Pause';
      timer = setInterval(() => {{
        let v = Number(stepInput.value);
        if (v >= maxStep) {{
          pause();
          return;
        }}
        stepInput.value = String(v + 1);
        render(v + 1);
      }}, Number(speedSel.value));
    }}
    function pause() {{
      if (timer) {{
        clearInterval(timer);
        timer = null;
      }}
      playBtn.textContent = 'Play';
    }}

    stepInput.addEventListener('input', () => render(Number(stepInput.value)));
    playBtn.addEventListener('click', () => (timer ? pause() : play()));
    prevBtn.addEventListener('click', () => {{
      const v = Math.max(0, Number(stepInput.value) - 1);
      stepInput.value = String(v);
      render(v);
    }});
    nextBtn.addEventListener('click', () => {{
      const v = Math.min(maxStep, Number(stepInput.value) + 1);
      stepInput.value = String(v);
      render(v);
    }});
    speedSel.addEventListener('change', () => {{
      if (timer) {{ pause(); play(); }}
    }});
    hideSpecial.addEventListener('change', () => render(Number(stepInput.value)));
    showMaskedShapes.addEventListener('change', () => render(Number(stepInput.value)));

    render(0);
  </script>
</body>
</html>"""

out_path = "trace_viewer.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"[Trace Viewer] Wrote: {os.path.abspath(out_path)}  (open in your browser)")



