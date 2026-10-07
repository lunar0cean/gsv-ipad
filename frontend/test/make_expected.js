// 用 Node 跑一遍 GSV.prepare，把结果存成模拟器测试的标准答案。
//   node make_expected.js <gsv-labels 可执行文件> <输出.json>
// 模拟器上的测试拿同样的句子再跑一遍，两边一致就说明：苹果的 JavaScriptCore 与 Node 对这些脚本的
// 执行结果相同，iOS 版的 Rust 库与本机版给出的日文标签相同。
const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");
const { loadFrontend } = require("./harness");

const [cli, out] = process.argv.slice(2);
const read = (name) => fs.readFileSync(path.join(__dirname, name), "utf8").split("\n").map((s) => s.trim()).filter(Boolean);
const cases = [
  ...read("ja_sentences.txt").map((text) => ({ text, lang: "ja" })),
  ...read("zh_sentences.txt").map((text) => ({ text, lang: "zh" })),
  { text: "第一行。\n第二行没有标点\n\n第三行，比较长的一句话，用来检查切句和停顿是不是一致。", lang: "zh" },
  { text: "一行目です。\n二行目は句読点なし\n三行目は、少し長めの文で、区切りと間の取り方を確かめます。", lang: "ja" },
];

// 第一遍只收集脚本会问到的日文片段，交给命令行工具一次算完；第二遍再用真标签
const wanted = new Set();
const first = loadFrontend((segment) => { wanted.add(segment); return ""; });
for (const item of cases.filter((c) => c.lang === "ja")) {
  first.GSV.prepare(item.text, item.lang);
}
const segments = [...wanted];
const run = spawnSync(cli, [], { input: segments.join("\n") + "\n", encoding: "utf8", maxBuffer: 1 << 28 });
if (run.status !== 0) {
  console.error("运行失败：", run.error || run.stderr);
  process.exit(2);
}
const blocks = run.stdout.split("\n@@\n");
const labels = new Map(segments.map((segment, index) => [segment, (blocks[index] || "").trim()]));

const { GSV } = loadFrontend((segment) => labels.get(segment) || "");
const expected = cases.map((item) => ({ ...item, segments: GSV.prepare(item.text, item.lang) }));
fs.writeFileSync(out, JSON.stringify(expected));
console.log(`标准答案 ${expected.length} 段文字 -> ${out}`);
