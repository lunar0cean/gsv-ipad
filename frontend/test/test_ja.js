// 日文对照测试。
//   node test_ja.js              用样本里原版 OpenJTalk 的标签，检验 JS 移植是否与原版逐音素一致（应当 100%）
//   node test_ja.js <gsv-labels> 用 Rust 库（jpreprocess）给出的标签，看它和原版 OpenJTalk 差多少
const { spawnSync } = require("child_process");
const path = require("path");
const { loadFrontend } = require("./harness");

const golden = require(path.join(__dirname, "ja_golden.json"));
const cli = process.argv[2];

let labelsBySegment = {};
for (const item of golden) {
  for (const [segment, labels] of Object.entries(item.segments)) {
    labelsBySegment[segment] = labels.join("\n");
  }
}

if (cli) {
  const segments = Object.keys(labelsBySegment);
  const run = spawnSync(cli, [], { input: segments.join("\n") + "\n", encoding: "utf8", maxBuffer: 1 << 28 });
  if (run.status !== 0) {
    console.error("运行失败：", run.error || run.stderr);
    process.exit(2);
  }
  const blocks = run.stdout.split("\n@@\n");
  labelsBySegment = {};
  segments.forEach((segment, index) => { labelsBySegment[segment] = (blocks[index] || "").trim(); });
}

const missing = new Set();
const { GSV } = loadFrontend((segment) => {
  if (!(segment in labelsBySegment)) {
    missing.add(segment);
    return "";
  }
  return labelsBySegment[segment];
});

let same = 0;
let phoneTotal = 0;
let phoneDiff = 0;
for (const item of golden) {
  const got = GSV.g2p(item.text, "ja").phones;
  const want = item.phones;
  phoneTotal += want.length;
  if (got.join(" ") === want.join(" ")) {
    same += 1;
    continue;
  }
  const length = Math.max(got.length, want.length);
  let diff = 0;
  for (let i = 0; i < length; i++) {
    if (got[i] !== want[i]) diff += 1;
  }
  phoneDiff += diff;
  console.log(`不一致：${item.text}\n  原版 ${want.join(" ")}\n  本次 ${got.join(" ")}`);
}
console.log(`\n${cli ? "jpreprocess 对照原版" : "JS 移植对照原版"}：${same}/${golden.length} 句完全一致，` +
  `不一致的句子里按位置比较有 ${phoneDiff}/${phoneTotal} 个音素不同`);
if (missing.size > 0) {
  console.log("样本里没有这些片段的标签：", [...missing]);
}
process.exit(!cli && same !== golden.length ? 1 : 0);
