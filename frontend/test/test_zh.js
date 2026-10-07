// 中文对照测试：zh.js 的规范化文本、分词（带词性）和音素应与原版完全一致。
//   node test_zh.js                 用仓库里的样本 zh_golden.json
//   node test_zh.js <golden.json>   用另外生成的样本（本机压力测试）
const path = require("path");
const { loadFrontend } = require("./harness");

const golden = require(path.resolve(process.argv[2] || path.join(__dirname, "zh_golden.json")));
const { GSV, GSV_ZH } = loadFrontend();

const started = Date.now();
GSV.g2p("你好", "zh");
const loadMs = Date.now() - started;

let same = 0;
let skipped = 0;
let shown = 0;
const failures = { norm: 0, words: 0, phones: 0 };
const begin = Date.now();
for (const item of golden) {
  if (item.error) {
    skipped += 1;
    continue;
  }
  let got;
  try {
    got = GSV.g2p(item.text, "zh");
  } catch (error) {
    got = { norm: "", phones: [], error: String(error && error.stack ? error.stack : error) };
  }
  if (got.norm === item.norm && got.phones.join(" ") === item.phones.join(" ")) {
    same += 1;
    continue;
  }
  const words = got.error ? [] : GSV_ZH.posCut(item.norm);
  const wordsText = (list) => list.map((w) => w.join("/")).join(" ");
  const kind = got.norm !== item.norm ? "norm" : (wordsText(words) !== wordsText(item.words) ? "words" : "phones");
  failures[kind] += 1;
  if (shown < 25) {
    shown += 1;
    console.log(`不一致（${{ norm: "规范化", words: "分词", phones: "读音" }[kind]}）：${item.text}`);
    if (got.error) {
      console.log(`  出错 ${got.error}`);
    } else if (kind === "norm") {
      console.log(`  原版 ${item.norm}\n  本次 ${got.norm}`);
    } else if (kind === "words") {
      console.log(`  原版 ${wordsText(item.words)}\n  本次 ${wordsText(words)}`);
    } else {
      console.log(`  分词 ${wordsText(words)}\n  原版 ${item.phones.join(" ")}\n  本次 ${got.phones.join(" ")}`);
    }
  }
}
const total = golden.length - skipped;
console.log(`\n中文 JS 移植对照原版：${same}/${total} 句完全一致` +
  `（规范化不同 ${failures.norm}，分词不同 ${failures.words}，读音不同 ${failures.phones}；原版自己报错而跳过 ${skipped}）`);
console.log(`载入词典 ${loadMs} 毫秒，处理 ${total} 句 ${Date.now() - begin} 毫秒`);
process.exit(same === total ? 0 : 1);
