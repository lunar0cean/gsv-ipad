// 切句测试：GSV.prepare 切出来的每个片段都必须有能读出来的内容。
// 起因：每行都用「」括起来的日文对话，行尾的「」」曾被切成一个只有标点的片段，
// 模型对它生成不出任何语义，整段合成就中断了。
//   node test_prepare.js
const path = require("path");
const { loadFrontend } = require("./harness");

const golden = require(path.join(__dirname, "ja_golden.json"));
const labels = {};
for (const item of golden) {
  for (const [segment, lines] of Object.entries(item.segments)) {
    labels[segment] = lines.join("\n");
  }
}
const { GSV } = loadFrontend((segment) => labels[segment] || "");

let failures = 0;
function check(name, condition, detail) {
  if (!condition) {
    failures += 1;
    console.log(`失败：${name}${detail ? "\n  " + detail : ""}`);
  }
}

const PUNCT = new Set(["!", "?", "…", ",", ".", "-", "UNK", "SP", "SP2", "SP3", "_"]);
const idToSymbol = Object.fromEntries(Object.entries(GSV.SYMBOL_ID).map(([symbol, id]) => [id, symbol]));
const speakable = (segment) => segment.ids.some((id) => !PUNCT.has(idToSymbol[id]));
const ids = (segments) => JSON.stringify(segments.map((s) => s.ids));

// 日文：样本句子逐行加上各种引号，结果应与不加引号时完全相同
const jaLines = golden.slice(0, 12).map((item) => item.text);
const jaPlain = GSV.prepare(jaLines.join("\n"), "ja");
for (const [open, close] of [["「", "」"], ["『", "』"], ["“", "”"], ['"', '"']]) {
  const quoted = GSV.prepare(jaLines.map((line) => open + line + close).join("\n"), "ja");
  check(`日文加 ${open}${close} 后片段不变`, ids(quoted) === ids(jaPlain), `片段数 ${quoted.length}，应为 ${jaPlain.length}`);
  check(`日文加 ${open}${close} 后每个片段都读得出来`, quoted.every(speakable));
}
check("日文样本切出了片段", jaPlain.length >= jaLines.length);

// 中文：同样的检查
const zhLines = [
  "你好。我是隔壁的佐藤。快递我先帮你收下了。",
  "啊，太谢谢了。我不在家，真不好意思。",
  "没关系。我正好在家，不麻烦的。",
  "这个周末你打算去哪里？",
];
const zhPlain = GSV.prepare(zhLines.join("\n"), "zh");
for (const [open, close] of [["“", "”"], ["「", "」"], ["‘", "’"]]) {
  const quoted = GSV.prepare(zhLines.map((line) => open + line + close).join("\n"), "zh");
  check(`中文加 ${open}${close} 后片段不变`, ids(quoted) === ids(zhPlain), `片段数 ${quoted.length}，应为 ${zhPlain.length}`);
  check(`中文加 ${open}${close} 后每个片段都读得出来`, quoted.every(speakable));
}

// 只有标点、符号或空行的输入不产生片段
for (const [text, lang] of [["。。。", "zh"], ["「」", "ja"], ["……！？", "zh"], ["\n\n  \n", "ja"], ["——", "zh"], ["（）", "zh"]]) {
  const segments = GSV.prepare(text, lang);
  check(`「${text.replace(/\n/g, "\\n")}」不产生片段`, segments.length === 0, JSON.stringify(segments.map((s) => s.text)));
}

// 没有标点收尾的行会补上句号，正常合成
check("没有标点的行仍然有片段", GSV.prepare("今天天气不错", "zh").length === 1);

console.log(failures === 0 ? "切句测试全部通过" : `切句测试有 ${failures} 项失败`);
process.exit(failures === 0 ? 0 : 1);
